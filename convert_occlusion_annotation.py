import io
import contextlib
import os
import datetime
import json
import numpy as np
from skimage import measure
from typing import List, Tuple, Union, Iterator
from pycocotools.coco import COCO
import pycocotools.mask as mask_utils
import torch
from enum import Enum
import math

from pycocotools.mask import frPyObjects, decode
from tqdm import tqdm


_RawBoxType = Union[List[float], Tuple[float, ...], torch.Tensor, np.ndarray]


class BoxMode(Enum):
    """
    Enum of different ways to represent a box.

    Attributes:

        XYXY_ABS: (x0, y0, x1, y1) in absolute floating points coordinates.
            The coordinates in range [0, width or height].
        XYWH_ABS: (x0, y0, w, h) in absolute floating points coordinates.
        XYXY_REL: (x0, y0, x1, y1) in range [0, 1]. They are relative to the size of the image.
        XYWH_REL: (x0, y0, w, h) in range [0, 1]. They are relative to the size of the image.
        XYWHA_ABS: (xc, yc, w, h, a) in absolute floating points coordinates.
            (xc, yc) is the center of the rotated box, and the angle a is in degrees ccw.
    """

    XYXY_ABS = 0
    XYWH_ABS = 1
    XYXY_REL = 2
    XYWH_REL = 3
    XYWHA_ABS = 4

    @staticmethod
    def convert(
        box: _RawBoxType, from_mode: "BoxMode", to_mode: "BoxMode"
    ) -> _RawBoxType:
        """
        Args:
            box: can be a k-tuple, k-list or an Nxk array/tensor, where k = 4 or 5
            from_mode, to_mode (BoxMode)

        Returns:
            The converted box of the same type.
        """
        if from_mode == to_mode:
            return box

        original_type = type(box)
        is_numpy = isinstance(box, np.ndarray)
        single_box = isinstance(box, (list, tuple))
        if single_box:
            assert len(box) == 4 or len(box) == 5, (
                "BoxMode.convert takes either a k-tuple/list or an Nxk array/tensor,"
                " where k == 4 or 5"
            )
            arr = torch.tensor(box)[None, :]
        else:
            # avoid modifying the input box
            if is_numpy:
                arr = torch.from_numpy(np.asarray(box)).clone()
            else:
                arr = box.clone()

        assert to_mode.value not in [
            BoxMode.XYXY_REL,
            BoxMode.XYWH_REL,
        ] and from_mode.value not in [
            BoxMode.XYXY_REL,
            BoxMode.XYWH_REL,
        ], "Relative mode not yet supported!"

        if from_mode == BoxMode.XYWHA_ABS and to_mode == BoxMode.XYXY_ABS:
            assert (
                arr.shape[-1] == 5
            ), "The last dimension of input shape must be 5 for XYWHA format"
            original_dtype = arr.dtype
            arr = arr.double()

            w = arr[:, 2]
            h = arr[:, 3]
            a = arr[:, 4]
            c = torch.abs(torch.cos(a * math.pi / 180.0))
            s = torch.abs(torch.sin(a * math.pi / 180.0))
            # This basically computes the horizontal bounding rectangle of the rotated box
            new_w = c * w + s * h
            new_h = c * h + s * w

            # convert center to top-left corner
            arr[:, 0] -= new_w / 2.0
            arr[:, 1] -= new_h / 2.0
            # bottom-right corner
            arr[:, 2] = arr[:, 0] + new_w
            arr[:, 3] = arr[:, 1] + new_h

            arr = arr[:, :4].to(dtype=original_dtype)
        else:
            if to_mode == BoxMode.XYXY_ABS and from_mode == BoxMode.XYWH_ABS:
                arr[:, 2] += arr[:, 0]
                arr[:, 3] += arr[:, 1]
            elif from_mode == BoxMode.XYXY_ABS and to_mode == BoxMode.XYWH_ABS:
                arr[:, 2] -= arr[:, 0]
                arr[:, 3] -= arr[:, 1]
            else:
                raise NotImplementedError(
                    "Conversion from BoxMode {} to {} is not supported yet".format(
                        from_mode, to_mode
                    )
                )

        if single_box:
            return original_type(arr.flatten())
        if is_numpy:
            return arr.numpy()
        else:
            return arr


def cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


class Boxes:
    """
    This structure stores a list of boxes as a Nx4 torch.Tensor.
    It supports some common methods about boxes
    (`area`, `clip`, `nonempty`, etc),
    and also behaves like a Tensor
    (support indexing, `to(device)`, `.device`, and iteration over all boxes)

    Attributes:
        tensor (torch.Tensor): float matrix of Nx4.
    """

    BoxSizeType = Union[List[int], Tuple[int, int]]

    def __init__(self, tensor: torch.Tensor):
        """
        Args:
            tensor (Tensor[float]): a Nx4 matrix.  Each row is (x1, y1, x2, y2).
        """
        device = (
            tensor.device if isinstance(tensor, torch.Tensor) else torch.device("cpu")
        )
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        if tensor.numel() == 0:
            tensor = torch.zeros(0, 4, dtype=torch.float32, device=device)
        assert tensor.dim() == 2 and tensor.size(-1) == 4, tensor.size()

        self.tensor = tensor

    def clone(self) -> "Boxes":
        """
        Clone the Boxes.

        Returns:
            Boxes
        """
        return Boxes(self.tensor.clone())

    def to(self, device: str) -> "Boxes":
        return Boxes(self.tensor.to(device))

    def area(self) -> torch.Tensor:
        """
        Computes the area of all the boxes.

        Returns:
            torch.Tensor: a vector with areas of each box.
        """
        box = self.tensor
        area = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
        return area

    def clip(self, box_size: BoxSizeType) -> None:
        """
        Clip (in place) the boxes by limiting x coordinates to the range [0, width]
        and y coordinates to the range [0, height].

        Args:
            box_size (height, width): The clipping box's size.
        """
        assert torch.isfinite(self.tensor).all(), "Box tensor contains infinite or NaN!"
        h, w = box_size
        self.tensor[:, 0].clamp_(min=0, max=w)
        self.tensor[:, 1].clamp_(min=0, max=h)
        self.tensor[:, 2].clamp_(min=0, max=w)
        self.tensor[:, 3].clamp_(min=0, max=h)

    def nonempty(self, threshold: int = 0) -> torch.Tensor:
        """
        Find boxes that are non-empty.
        A box is considered empty, if either of its side is no larger than threshold.

        Returns:
            Tensor:
                a binary vector which represents whether each box is empty
                (False) or non-empty (True).
        """
        box = self.tensor
        widths = box[:, 2] - box[:, 0]
        heights = box[:, 3] - box[:, 1]
        keep = (widths > threshold) & (heights > threshold)
        return keep

    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "Boxes":
        """
        Returns:
            Boxes: Create a new :class:`Boxes` by indexing.

        The following usage are allowed:
        1. `new_boxes = boxes[3]`: return a `Boxes` which contains only one box.
        2. `new_boxes = boxes[2:10]`: return a slice of boxes.
        3. `new_boxes = boxes[vector]`, where vector is a torch.BoolTensor
           with `length = len(boxes)`. Nonzero elements in the vector will be selected.

        Note that the returned Boxes might share storage with this Boxes,
        subject to Pytorch's indexing semantics.
        """
        if isinstance(item, int):
            return Boxes(self.tensor[item].view(1, -1))
        b = self.tensor[item]
        assert (
            b.dim() == 2
        ), "Indexing on Boxes with {} failed to return a matrix!".format(item)
        return Boxes(b)

    def __len__(self) -> int:
        return self.tensor.shape[0]

    def __repr__(self) -> str:
        return "Boxes(" + str(self.tensor) + ")"

    def inside_box(
        self, box_size: BoxSizeType, boundary_threshold: int = 0
    ) -> torch.Tensor:
        """
        Args:
            box_size (height, width): Size of the reference box.
            boundary_threshold (int): Boxes that extend beyond the reference box
                boundary by more than boundary_threshold are considered "outside".

        Returns:
            a binary vector, indicating whether each box is inside the reference box.
        """
        height, width = box_size
        inds_inside = (
            (self.tensor[..., 0] >= -boundary_threshold)
            & (self.tensor[..., 1] >= -boundary_threshold)
            & (self.tensor[..., 2] < width + boundary_threshold)
            & (self.tensor[..., 3] < height + boundary_threshold)
        )
        return inds_inside

    def get_centers(self) -> torch.Tensor:
        """
        Returns:
            The box centers in a Nx2 array of (x, y).
        """
        return (self.tensor[:, :2] + self.tensor[:, 2:]) / 2

    def scale(self, scale_x: float, scale_y: float) -> None:
        """
        Scale the box with horizontal and vertical scaling factors
        """
        self.tensor[:, 0::2] *= scale_x
        self.tensor[:, 1::2] *= scale_y

    @staticmethod
    def cat(boxes_list: List["Boxes"]) -> "Boxes":
        """
        Concatenates a list of Boxes into a single Boxes

        Arguments:
            boxes_list (list[Boxes])

        Returns:
            Boxes: the concatenated Boxes
        """
        assert isinstance(boxes_list, (list, tuple))
        assert len(boxes_list) > 0
        assert all(isinstance(box, Boxes) for box in boxes_list)

        cat_boxes = type(boxes_list[0])(cat([b.tensor for b in boxes_list], dim=0))
        return cat_boxes

    @property
    def device(self) -> torch.device:
        return self.tensor.device

    def __iter__(self) -> Iterator[torch.Tensor]:
        """
        Yield a box as a Tensor of shape (4,) at a time.
        """
        yield from self.tensor


def polygons_to_bitmask(
    polygons: List[np.ndarray], height: int, width: int
) -> np.ndarray:
    assert len(polygons) > 0, "COCOAPI does not support empty polygons"
    rles = mask_utils.frPyObjects(polygons, height, width)
    rle = mask_utils.merge(rles)
    return mask_utils.decode(rle).astype(np.bool_)


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def load_coco_json(json_file, image_root):
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)

    img_ids = sorted(coco_api.imgs.keys())

    imgs = coco_api.loadImgs(img_ids)

    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]

    imgs_anns = list(zip(imgs, anns))

    categories = coco_api.loadCats(coco_api.getCatIds())

    dataset_dicts = []

    ann_keys = ["iscrowd", "bbox", "keypoints", "category_id"]

    sum_box = 0
    sum_co_box = 0
    intersect_rate = 0.0
    intersect_num = 0

    for _, (img_dict, anno_dict_list) in enumerate(tqdm(imgs_anns)):
        record = {}
        record["file_name"] = os.path.join(img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]
        objs = []
        for anno in anno_dict_list:
            assert anno["image_id"] == image_id

            obj = {key: anno[key] for key in ann_keys if key in anno}

            segm = anno.get("segmentation", None)
            if segm:  # either list[list[float]] or dict(RLE)
                if not isinstance(segm, dict):
                    # filter out invalid polygons (< 3 points)
                    segm = [
                        poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6
                    ]
                obj["segmentation"] = segm

            keypts = anno.get("keypoints", None)
            if keypts:  # list[int]
                for idx, v in enumerate(keypts):
                    if idx % 3 != 2:
                        keypts[idx] = v + 0.5
                obj["keypoints"] = keypts

            obj["bbox_mode"] = BoxMode.XYWH_ABS
            objs.append(obj)
        record["annotations"] = objs
        seg_list = []
        for obj in objs:
            seg_list.append(obj["segmentation"])

        bitmask_list = []
        if len(seg_list) > 0:
            for index, seg in enumerate(seg_list):
                # print('seg len:', len(seg))
                invalid = False
                for sub_seg in seg:
                    # print('seg len:', len(sub_seg))
                    if len(sub_seg) < 6:
                        invalid = True
                if not invalid:
                    bitmask = polygons_to_bitmask(
                        seg, img_dict["height"], img_dict["width"]
                    )
                else:
                    bitmask = np.zeros(
                        (int(img_dict["height"]), int(img_dict["width"])), dtype=bool
                    )

                bitmask_list.append(bitmask.astype("int"))

        box_list = []
        for obj in objs:
            box_list.append(
                [
                    obj["bbox"][0],
                    obj["bbox"][1],
                    obj["bbox"][0] + obj["bbox"][2],
                    obj["bbox"][1] + obj["bbox"][3],
                ]
            )

        box_mask_list = []
        for index, obj in enumerate(objs):
            box_mask = np.zeros(
                (int(img_dict["height"]), int(img_dict["width"])), dtype=int
            )
            box_mask[
                int(box_list[index][1]) : int(box_list[index][3]),
                int(box_list[index][0]) : int(box_list[index][2]),
            ] = 1
            box_mask_list.append(box_mask)

        sum_box += len(box_list)

        for index1, a_box in enumerate(box_list):
            union_mask_whole = np.zeros(
                (int(img_dict["height"]), int(img_dict["width"])), dtype=int
            )
            for index2, b_box in enumerate(box_list):
                if index1 != index2:
                    iou = bb_intersection_over_union(a_box, b_box)
                    if iou > 0.05:
                        union_mask = np.multiply(
                            box_mask_list[index1], bitmask_list[index2]
                        )
                        union_mask_whole += union_mask

            union_mask_whole[union_mask_whole > 1.0] = 1.0
            intersect_mask = union_mask_whole * bitmask_list[index1]

            if intersect_mask.sum() >= 1.0:
                intersect_num += 1

            if float(bitmask_list[index1].sum()) > 1.0:
                intersect_rate += intersect_mask.sum() / float(
                    bitmask_list[index1].sum()
                )

            union_mask_non_zero_num = np.count_nonzero(union_mask_whole.astype(int))
            record["annotations"][index1]["bg_object_segmentation"] = []
            if union_mask_non_zero_num > 20:
                sum_co_box += 1
                contours = measure.find_contours(union_mask_whole.astype(int), 0)
                for contour in contours:
                    if contour.shape[0] > 500:
                        contour = np.flip(contour, axis=1)[::10, :]
                    elif contour.shape[0] > 200:
                        contour = np.flip(contour, axis=1)[::5, :]
                    elif contour.shape[0] > 100:
                        contour = np.flip(contour, axis=1)[::3, :]
                    elif contour.shape[0] > 50:
                        contour = np.flip(contour, axis=1)[::2, :]
                    else:
                        contour = np.flip(contour, axis=1)

                    segmentation = contour.ravel().tolist()
                    record["annotations"][index1]["bg_object_segmentation"].append(
                        segmentation
                    )

        dataset_dicts.append(record)

    return dataset_dicts, categories


def convert_to_coco_dict(dataset_dicts, categories):
    coco_images = []
    coco_annotations = []

    for image_id, image_dict in enumerate(tqdm(dataset_dicts)):
        coco_image = {
            "id": image_dict.get("image_id", image_id),
            "width": image_dict["width"],
            "height": image_dict["height"],
            "file_name": image_dict["file_name"],
        }
        coco_images.append(coco_image)

        anns_per_image = image_dict["annotations"]
        for annotation in anns_per_image:
            # create a new dict with only COCO fields
            coco_annotation = {}

            # COCO requirement: XYWH box format
            bbox = annotation["bbox"]
            bbox_mode = annotation["bbox_mode"]
            bbox = BoxMode.convert(bbox, bbox_mode, BoxMode.XYWH_ABS)
            # print('annotation:', annotation)
            # COCO requirement: instance area
            if "segmentation" in annotation:
                # Computing areas for instances by counting the pixels
                # TODO: check segmentation type: RLE, BinaryMask or Polygon
                # if len(segmentation) >= 1:
                #    polygons = PolygonMasks([segmentation])
                #    area = polygons.area()[0].item()
                # else:
                bbox_xy = BoxMode.convert(bbox, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
                area = Boxes([bbox_xy]).area()[0].item()
            else:
                # Computing areas using bounding boxes
                bbox_xy = BoxMode.convert(bbox, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
                area = Boxes([bbox_xy]).area()[0].item()

            if "keypoints" in annotation:
                keypoints = annotation["keypoints"]  # list[int]
                for idx, v in enumerate(keypoints):
                    if idx % 3 != 2:
                        # COCO's segmentation coordinates are floating points in [0, H or W],
                        # but keypoint coordinates are integers in [0, H-1 or W-1]
                        # For COCO format consistency we substract 0.5
                        # https://github.com/facebookresearch/detectron2/pull/175#issuecomment-551202163
                        keypoints[idx] = v - 0.5
                if "num_keypoints" in annotation:
                    num_keypoints = annotation["num_keypoints"]
                else:
                    num_keypoints = sum(kp > 0 for kp in keypoints[2::3])

            # COCO requirement:
            #   linking annotations to images
            #   "id" field must start with 1
            coco_annotation["id"] = len(coco_annotations) + 1
            coco_annotation["image_id"] = coco_image["id"]
            coco_annotation["bbox"] = [round(float(x), 3) for x in bbox]
            coco_annotation["area"] = area
            coco_annotation["iscrowd"] = annotation.get("iscrowd", 0)
            coco_annotation["category_id"] = annotation["category_id"]

            # Add optional fields
            if "keypoints" in annotation:
                coco_annotation["keypoints"] = keypoints
                coco_annotation["num_keypoints"] = num_keypoints

            if "segmentation" in annotation:
                coco_annotation["segmentation"] = annotation["segmentation"]
            if "bg_object_segmentation" in annotation:
                coco_annotation["bg_object_segmentation"] = annotation[
                    "bg_object_segmentation"
                ]

            coco_annotations.append(coco_annotation)

    info = {
        "date_created": str(datetime.datetime.now()),
        "description": "Automatically generated COCO json file for Detectron2.",
    }
    coco_dict = {
        "info": info,
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": categories,
        "licenses": None,
    }
    return coco_dict


def polygon_to_mask(polygon, height, width):
    rle = frPyObjects(polygon, height, width)
    mask = decode(rle)
    return mask


def check_overlap(mask1, mask2):
    overlap = (mask1 & mask2).sum() > 0
    return overlap


def statics(file):
    with open(file) as f:
        more_annos = []
        coco = COCO(file)
        data = json.load(f)
        bo_count = 0
        img_bo_ids = []
        img_ids = []
        for ann in data["annotations"]:
            if "bg_object_segmentation" in ann and ann["bg_object_segmentation"] != []:
                bo_count += 1
                img_bo_ids.append(ann["image_id"])

            if "segmentation" in ann and ann["segmentation"] != []:
                img_ids.append(ann["image_id"])

        img_ids = list(set(img_ids))

        img_bo_ids = list(set(img_bo_ids))

        annotated_image_ids = set([ann["image_id"] for ann in data["annotations"]])

        count = 0
        for image_id in annotated_image_ids:
            img_info = coco.loadImgs(image_id)[0]
            annIds = coco.getAnnIds(imgIds=img_info["id"], iscrowd=None)
            if len(annIds) > 1:
                count += 1
        more_annos.append(count)

        print(f'images count: {len(data["images"])}')
        print(f"images with segmentation count: {len(img_ids)}")
        print(
            f"Total number of images with more than one annotation: {sum(more_annos)}"
        )
        print(f"images with bg_object_segmentation count: {len(img_bo_ids)}")

        print(f'annotations count: {len(data["annotations"])}')
        print(f"bg_object_segmentation count: {bo_count}")


if __name__ == "__main__":
    # The paths of train and test json files of your own COCO format dataset
    file_lst = [
        "xxx/annotations/train.json",
        "xxx/annotations/test.json",
    ]

    # The paths of train and test image folders of your own COCO format dataset
    for json_file in file_lst:
        if json_file == "train.json":
            image_root = "xxx/train"
        else:
            image_root = "xxx/test"

        dicts, categories = load_coco_json(json_file, image_root)
        coco_dict = convert_to_coco_dict(dicts, categories)

        output_file = json_file.replace(".json", "_occ.json")

        problematic_annotations = []

        for ann in tqdm(coco_dict["annotations"]):
            segmentation = ann["segmentation"]
            bg_segmentation = ann.get("bg_object_segmentation", [])

            if bg_segmentation != []:
                img_info = next(
                    (
                        img
                        for img in coco_dict["images"]
                        if img["id"] == ann["image_id"]
                    ),
                    None,
                )
                height, width = img_info["height"], img_info["width"]

                seg_mask = polygon_to_mask(segmentation, height, width)
                bg_seg_mask = polygon_to_mask(bg_segmentation, height, width)

                if not check_overlap(seg_mask, bg_seg_mask):
                    problematic_annotations.append(ann["id"])

        for ann in coco_dict["annotations"]:
            if ann["id"] in problematic_annotations:
                ann["bg_object_segmentation"] = []

        with open(output_file, "w") as f:
            json.dump(coco_dict, f)

        statics(output_file)
