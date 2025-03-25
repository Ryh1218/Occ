import os.path as osp
from typing import List, Optional, Sequence, Tuple, Union

import einops
import mmcv
import numpy as np
import torch
from mmcv.image.geometric import _scale_size
from mmcv.transforms import to_tensor
from mmengine import ConfigDict
from mmengine.model import BaseModule
from mmengine.structures import InstanceData, PixelData
from torch import Tensor
from torch.nn import functional as F
from transformers import SamConfig
from transformers.models.sam.modeling_sam import SamMaskDecoder

from mmdet.datasets.coco import CocoDataset
from mmdet.datasets.transforms.formatting import PackDetInputs
from mmdet.datasets.transforms.loading import LoadAnnotations
from mmdet.datasets.transforms.transforms import RandomCrop, RandomFlip, Resize
from mmdet.models.data_preprocessors import BatchFixedSizePad, DetDataPreprocessor
from mmdet.models.task_modules import SamplingResult
from mmdet.models.utils import unpack_gt_instances
from mmdet.models.utils.misc import samplelist_boxtype2tensor
from mmdet.occlu.models import (
    RSPrompterAnchorMaskHead,
    RSPrompterAnchorRoIPromptHead,
    RSSamMaskDecoder,
)
from mmdet.registry import DATASETS, MODELS, TRANSFORMS
from mmdet.structures import DetDataSample
from mmdet.structures.bbox import BaseBoxes, autocast_box_type, bbox2roi
from mmdet.structures.mask import BitmapMasks, PolygonMasks
from mmdet.utils import InstanceList


@DATASETS.register_module()
class OcCocoDataset(CocoDataset):
    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        img_info = raw_data_info["raw_img_info"]
        ann_info = raw_data_info["raw_ann_info"]

        data_info = {}

        # TODO: need to change data_prefix['img'] to data_prefix['img_path']
        img_path = osp.join(self.data_prefix["img"], img_info["file_name"])
        if self.data_prefix.get("seg", None):
            seg_map_path = osp.join(
                self.data_prefix["seg"],
                img_info["file_name"].rsplit(".", 1)[0] + self.seg_map_suffix,
            )
        else:
            seg_map_path = None
        data_info["img_path"] = img_path
        data_info["img_id"] = img_info["img_id"]
        data_info["seg_map_path"] = seg_map_path
        data_info["height"] = img_info["height"]
        data_info["width"] = img_info["width"]

        if self.return_classes:
            data_info["text"] = self.metainfo["classes"]
            data_info["custom_entities"] = True

        instances = []
        for i, ann in enumerate(ann_info):
            instance = {}

            if ann.get("ignore", False):
                continue
            x1, y1, w, h = ann["bbox"]
            inter_w = max(0, min(x1 + w, img_info["width"]) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info["height"]) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann["area"] <= 0 or w < 1 or h < 1:
                continue
            if ann["category_id"] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]

            if ann.get("iscrowd", False):
                instance["ignore_flag"] = 1
            else:
                instance["ignore_flag"] = 0
            instance["bbox"] = bbox
            instance["bbox_label"] = self.cat2label[ann["category_id"]]

            if ann.get("segmentation", None):
                instance["mask"] = ann["segmentation"]
                instance["bo_mask"] = ann["bg_object_segmentation"]

            instances.append(instance)
        data_info["instances"] = instances
        return data_info


@TRANSFORMS.register_module()
class OcLoadAnnotations(LoadAnnotations):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _process_masks(self, results: dict) -> list:
        gt_masks = []
        gt_ignore_flags = []
        for instance in results.get("instances", []):
            gt_mask = instance["mask"]
            # If the annotation of segmentation mask is invalid,
            # ignore the whole instance.
            if isinstance(gt_mask, list):
                gt_mask = [
                    np.array(polygon)
                    for polygon in gt_mask
                    if len(polygon) % 2 == 0 and len(polygon) >= 6
                ]
                if len(gt_mask) == 0:
                    # ignore this instance and set gt_mask to a fake mask
                    instance["ignore_flag"] = 1
                    gt_mask = [np.zeros(6)]
            elif not self.poly2mask:
                # `PolygonMasks` requires a ploygon of format List[np.array],
                # other formats are invalid.
                instance["ignore_flag"] = 1
                gt_mask = [np.zeros(6)]
            elif isinstance(gt_mask, dict) and not (
                gt_mask.get("counts") is not None
                and gt_mask.get("size") is not None
                and isinstance(gt_mask["counts"], (list, str))
            ):
                # if gt_mask is a dict, it should include `counts` and `size`,
                # so that `BitmapMasks` can uncompressed RLE
                instance["ignore_flag"] = 1
                gt_mask = [np.zeros(6)]
            gt_masks.append(gt_mask)
            # re-process gt_ignore_flags
            gt_ignore_flags.append(instance["ignore_flag"])
        results["gt_ignore_flags"] = np.array(gt_ignore_flags, dtype=bool)
        return gt_masks

    def _process_bo_masks(self, results: dict) -> list:
        gt_bo_masks = []
        for instance in results.get("instances", []):
            gt_bo_mask = instance["bo_mask"]
            if isinstance(gt_bo_mask, list):
                gt_bo_mask = [
                    np.array(polygon)
                    for polygon in gt_bo_mask
                    if len(polygon) % 2 == 0 and len(polygon) >= 6
                ]
                if len(gt_bo_mask) == 0:
                    gt_bo_mask = [np.zeros(6)]
            elif not self.poly2mask:
                gt_bo_mask = [np.zeros(6)]
            gt_bo_masks.append(gt_bo_mask)
        return gt_bo_masks

    def _load_masks(self, results: dict) -> None:
        h, w = results["ori_shape"]
        gt_masks = self._process_masks(results)
        gt_bo_masks = self._process_bo_masks(results)
        if self.poly2mask:
            gt_masks = BitmapMasks(
                [self._poly2mask(mask, h, w) for mask in gt_masks], h, w
            )
            gt_bo_masks = BitmapMasks(
                [self._poly2mask(mask, h, w) for mask in gt_bo_masks], h, w
            )
        else:
            # fake polygon masks will be ignored in `PackDetInputs`
            gt_masks = PolygonMasks([mask for mask in gt_masks], h, w)
            gt_bo_masks = PolygonMasks([mask for mask in gt_bo_masks], h, w)
        results["gt_masks"] = gt_masks
        results["gt_bo_masks"] = gt_bo_masks

    def transform(self, results: dict) -> dict:
        if self.with_bbox:
            self._load_bboxes(results)
        if self.with_label:
            self._load_labels(results)
        if self.with_mask:
            self._load_masks(results)
        if self.with_seg:
            self._load_seg_map(results)
        return results


@TRANSFORMS.register_module()
class OcRandomFlip(RandomFlip):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @autocast_box_type()
    def _flip(self, results: dict) -> None:
        """Flip images, bounding boxes, and semantic segmentation map."""
        # flip image
        results["img"] = mmcv.imflip(
            results["img"], direction=results["flip_direction"]
        )

        img_shape = results["img"].shape[:2]

        # flip bboxes
        if results.get("gt_bboxes", None) is not None:
            results["gt_bboxes"].flip_(img_shape, results["flip_direction"])

        # flip masks
        if results.get("gt_masks", None) is not None:
            results["gt_masks"] = results["gt_masks"].flip(results["flip_direction"])
            results["gt_bo_masks"] = results["gt_bo_masks"].flip(
                results["flip_direction"]
            )

        # flip segs
        if results.get("gt_seg_map", None) is not None:
            results["gt_seg_map"] = mmcv.imflip(
                results["gt_seg_map"], direction=results["flip_direction"]
            )

        self._record_homography_matrix(results)


@TRANSFORMS.register_module()
class OcResize(Resize):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _resize_masks(self, results: dict) -> None:
        """Resize masks with ``results['scale']``"""
        if results.get("gt_masks", None) is not None:
            if self.keep_ratio:
                results["gt_masks"] = results["gt_masks"].rescale(results["scale"])
                results["gt_bo_masks"] = results["gt_bo_masks"].rescale(
                    results["scale"]
                )
            else:
                results["gt_masks"] = results["gt_masks"].resize(results["img_shape"])
                results["gt_bo_masks"] = results["gt_bo_masks"].resize(
                    results["img_shape"]
                )

    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        if self.scale:
            results["scale"] = self.scale
        else:
            img_shape = results["img"].shape[:2]
            results["scale"] = _scale_size(img_shape[::-1], self.scale_factor)
        self._resize_img(results)
        self._resize_bboxes(results)
        self._resize_masks(results)
        self._resize_seg(results)
        self._record_homography_matrix(results)

        return results


@TRANSFORMS.register_module()
class OcRandomCrop(RandomCrop):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _crop_data(
        self, results: dict, crop_size: Tuple[int, int], allow_negative_crop: bool
    ) -> Union[dict, None]:
        assert crop_size[0] > 0 and crop_size[1] > 0
        img = results["img"]
        margin_h = max(img.shape[0] - crop_size[0], 0)
        margin_w = max(img.shape[1] - crop_size[1], 0)
        offset_h, offset_w = self._rand_offset((margin_h, margin_w))
        crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]

        # Record the homography matrix for the RandomCrop
        homography_matrix = np.array(
            [[1, 0, -offset_w], [0, 1, -offset_h], [0, 0, 1]], dtype=np.float32
        )
        if results.get("homography_matrix", None) is None:
            results["homography_matrix"] = homography_matrix
        else:
            results["homography_matrix"] = (
                homography_matrix @ results["homography_matrix"]
            )

        # crop the image
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        img_shape = img.shape
        results["img"] = img
        results["img_shape"] = img_shape[:2]

        # crop bboxes accordingly and clip to the image boundary
        if results.get("gt_bboxes", None) is not None:
            bboxes = results["gt_bboxes"]
            bboxes.translate_([-offset_w, -offset_h])
            if self.bbox_clip_border:
                bboxes.clip_(img_shape[:2])
            valid_inds = bboxes.is_inside(img_shape[:2]).numpy()
            # If the crop does not contain any gt-bbox area and
            # allow_negative_crop is False, skip this image.
            if not valid_inds.any() and not allow_negative_crop:
                return None

            results["gt_bboxes"] = bboxes[valid_inds]

            if results.get("gt_ignore_flags", None) is not None:
                results["gt_ignore_flags"] = results["gt_ignore_flags"][valid_inds]

            if results.get("gt_bboxes_labels", None) is not None:
                results["gt_bboxes_labels"] = results["gt_bboxes_labels"][valid_inds]

            if results.get("gt_masks", None) is not None:
                results["gt_masks"] = results["gt_masks"][valid_inds.nonzero()[0]].crop(
                    np.asarray([crop_x1, crop_y1, crop_x2, crop_y2])
                )
                # results["gt_bo_masks"] = results["gt_bo_masks"][
                #     valid_inds.nonzero()[0]
                # ].crop(np.asarray([crop_x1, crop_y1, crop_x2, crop_y2]))
                if self.recompute_bbox:
                    results["gt_bboxes"] = results["gt_masks"].get_bboxes(
                        type(results["gt_bboxes"])
                    )

                if results.get("gt_bo_masks", None) is not None:
                    results["gt_bo_masks"] = results["gt_bo_masks"][
                        valid_inds.nonzero()[0]
                    ].crop(np.asarray([crop_x1, crop_y1, crop_x2, crop_y2]))

            # We should remove the instance ids corresponding to invalid boxes.
            if results.get("gt_instances_ids", None) is not None:
                results["gt_instances_ids"] = results["gt_instances_ids"][valid_inds]

        # crop semantic seg
        if results.get("gt_seg_map", None) is not None:
            results["gt_seg_map"] = results["gt_seg_map"][
                crop_y1:crop_y2, crop_x1:crop_x2
            ]

        return results

    @autocast_box_type()
    def transform(self, results: dict) -> Union[dict, None]:
        image_size = results["img"].shape[:2]
        crop_size = self._get_crop_size(image_size)
        results = self._crop_data(results, crop_size, self.allow_negative_crop)

        return results


@TRANSFORMS.register_module()
class OcPackDetInputs(PackDetInputs):
    mapping_table = {
        "gt_bboxes": "bboxes",
        "gt_bboxes_labels": "labels",
        "gt_masks": "masks",
        "gt_bo_masks": "bo_masks",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def transform(self, results: dict) -> dict:
        packed_results = dict()
        if "img" in results:
            img = results["img"]
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            if not img.flags.c_contiguous:
                img = np.ascontiguousarray(img.transpose(2, 0, 1))
                img = to_tensor(img)
            else:
                img = to_tensor(img).permute(2, 0, 1).contiguous()

            packed_results["inputs"] = img

        if "gt_ignore_flags" in results:
            valid_idx = np.where(results["gt_ignore_flags"] == 0)[0]
            ignore_idx = np.where(results["gt_ignore_flags"] == 1)[0]

        data_sample = DetDataSample()
        instance_data = InstanceData()
        ignore_instance_data = InstanceData()

        for key in self.mapping_table.keys():
            if key not in results:
                continue
            if key == "gt_masks":
                if "gt_ignore_flags" in results:
                    instance_data[self.mapping_table[key]] = results[key][valid_idx]
                    ignore_instance_data[self.mapping_table[key]] = results[key][
                        ignore_idx
                    ]
                else:
                    instance_data[self.mapping_table[key]] = results[key]
            elif key == "gt_bo_masks":
                if "gt_ignore_flags" in results:
                    if len(results[key]) == len(valid_idx):
                        instance_data[self.mapping_table[key]] = results[key][valid_idx]
                        ignore_instance_data[self.mapping_table[key]] = results[key][
                            ignore_idx
                        ]
                    else:
                        instance_data[self.mapping_table[key]] = results["gt_masks"][
                            valid_idx
                        ]
                        ignore_instance_data[self.mapping_table[key]] = results[
                            "gt_masks"
                        ][ignore_idx]
                else:
                    instance_data[self.mapping_table[key]] = results[key]
            elif isinstance(results[key], BaseBoxes):
                if "gt_ignore_flags" in results:
                    instance_data[self.mapping_table[key]] = results[key][valid_idx]
                    ignore_instance_data[self.mapping_table[key]] = results[key][
                        ignore_idx
                    ]
                else:
                    instance_data[self.mapping_table[key]] = results[key]
            else:
                if "gt_ignore_flags" in results:
                    instance_data[self.mapping_table[key]] = to_tensor(
                        results[key][valid_idx]
                    )
                    ignore_instance_data[self.mapping_table[key]] = to_tensor(
                        results[key][ignore_idx]
                    )
                else:
                    instance_data[self.mapping_table[key]] = to_tensor(results[key])
        data_sample.gt_instances = instance_data
        data_sample.ignored_instances = ignore_instance_data

        if "proposals" in results:
            proposals = InstanceData(
                bboxes=to_tensor(results["proposals"]),
                scores=to_tensor(results["proposals_scores"]),
            )
            data_sample.proposals = proposals

        if "gt_seg_map" in results:
            gt_sem_seg_data = dict(
                sem_seg=to_tensor(results["gt_seg_map"][None, ...].copy())
            )
            gt_sem_seg_data = PixelData(**gt_sem_seg_data)
            if "ignore_index" in results:
                metainfo = dict(ignore_index=results["ignore_index"])
                gt_sem_seg_data.set_metainfo(metainfo)
            data_sample.gt_sem_seg = gt_sem_seg_data

        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
        data_sample.set_metainfo(img_meta)
        packed_results["data_samples"] = data_sample

        return packed_results


@MODELS.register_module()
class OcDataPreprocessor(DetDataPreprocessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def pad_gt_masks(self, batch_data_samples: Sequence[DetDataSample]) -> None:
        """Pad gt_masks to shape of batch_input_shape."""
        if "masks" in batch_data_samples[0].gt_instances:
            for data_samples in batch_data_samples:
                masks = data_samples.gt_instances.masks
                data_samples.gt_instances.masks = masks.pad(
                    data_samples.batch_input_shape, pad_val=self.mask_pad_value
                )
                bo_masks = data_samples.gt_instances.bo_masks
                data_samples.gt_instances.bo_masks = bo_masks.pad(
                    data_samples.batch_input_shape, pad_val=self.mask_pad_value
                )

    def forward(self, data: dict, training: bool = False) -> dict:
        batch_pad_shape = self._get_pad_shape(data)
        data = super().forward(data=data, training=training)
        inputs, data_samples = data["inputs"], data["data_samples"]

        if data_samples is not None:
            batch_input_shape = tuple(inputs[0].size()[-2:])
            for data_sample, pad_shape in zip(data_samples, batch_pad_shape):
                data_sample.set_metainfo(
                    {"batch_input_shape": batch_input_shape, "pad_shape": pad_shape}
                )

            if self.boxtype2tensor:
                samplelist_boxtype2tensor(data_samples)

            if self.pad_mask and training:
                self.pad_gt_masks(data_samples)

            if self.pad_seg and training:
                self.pad_gt_sem_seg(data_samples)

        if training and self.batch_augments is not None:
            for batch_aug in self.batch_augments:
                inputs, data_samples = batch_aug(inputs, data_samples)

        return {"inputs": inputs, "data_samples": data_samples}


@MODELS.register_module()
class OcBatchFixedSizePad(BatchFixedSizePad):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self, inputs: Tensor, data_samples: Optional[List[dict]] = None
    ) -> Tuple[Tensor, Optional[List[dict]]]:
        """Pad image, instance masks, segmantic segmentation maps."""
        src_h, src_w = inputs.shape[-2:]
        dst_h, dst_w = self.size

        if src_h >= dst_h and src_w >= dst_w:
            return inputs, data_samples

        inputs = F.pad(
            inputs,
            pad=(0, max(0, dst_w - src_w), 0, max(0, dst_h - src_h)),
            mode="constant",
            value=self.img_pad_value,
        )

        if data_samples is not None:
            # update batch_input_shape
            for data_sample in data_samples:
                data_sample.set_metainfo(
                    {"batch_input_shape": (dst_h, dst_w), "pad_shape": (dst_h, dst_w)}
                )

            if self.pad_mask:
                for data_sample in data_samples:
                    masks = data_sample.gt_instances.masks
                    data_sample.gt_instances.masks = masks.pad(
                        (dst_h, dst_w), pad_val=self.mask_pad_value
                    )
                    bo_masks = data_sample.gt_instances.bo_masks
                    data_sample.gt_instances.bo_masks = bo_masks.pad(
                        (dst_h, dst_w), pad_val=self.mask_pad_value
                    )

            if self.pad_seg:
                for data_sample in data_samples:
                    gt_sem_seg = data_sample.gt_sem_seg.sem_seg
                    h, w = gt_sem_seg.shape[-2:]
                    gt_sem_seg = F.pad(
                        gt_sem_seg,
                        pad=(0, max(0, dst_w - w), 0, max(0, dst_h - h)),
                        mode="constant",
                        value=self.seg_pad_value,
                    )
                    data_sample.gt_sem_seg = PixelData(sem_seg=gt_sem_seg)

        return inputs, data_samples


@MODELS.register_module()
class OcFeatureAggregator(BaseModule):
    def __init__(
        self,
        in_channels,
        hidden_channels=64,
        out_channels=256,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)

    def forward(self, inputs):
        # inputs: tuple, element: torch.Size([2, 256, 32, 32])
        assert len(inputs) == 1
        x = inputs[0]

        return x


@MODELS.register_module()
class OcPrompterAnchorRoIPromptHead(RSPrompterAnchorRoIPromptHead):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _mask_forward(
        self,
        x: Tuple[Tensor],
        rois: Tensor = None,
        pos_inds: Optional[Tensor] = None,
        bbox_feats: Optional[Tensor] = None,
        image_embeddings=None,
        image_positional_embeddings=None,
    ) -> dict:
        assert (rois is not None) ^ (pos_inds is not None and bbox_feats is not None)
        if rois is not None:
            mask_feats = self.mask_roi_extractor(
                x[: self.mask_roi_extractor.num_inputs], rois
            )
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
        else:
            assert bbox_feats is not None
            mask_feats = bbox_feats[pos_inds]

        # mask_feats shape [num_pos, 256, 14, 14]
        mask_feats_bo = mask_feats.clone()

        mask_preds_bo, iou_predictions, img_embeddings, point_embeddings = (
            self.mask_head(  # torch.Size([n, 1, 128, 128])
                mask_feats_bo,
                image_embeddings=image_embeddings,
                image_positional_embeddings=image_positional_embeddings,
                roi_img_ids=rois[:, 0] if rois is not None else None,
                if_bo=True,
                res_img_embeddings=None,
                res_point_embeddings=None,
            )
        )

        mask_preds, iou_predictions, _, _ = (
            self.mask_head(  # torch.Size([n, 1, 128, 128])
                mask_feats,
                image_embeddings=image_embeddings,
                image_positional_embeddings=image_positional_embeddings,
                roi_img_ids=rois[:, 0] if rois is not None else None,
                if_bo=False,
                res_img_embeddings=img_embeddings,
                res_point_embeddings=point_embeddings,
            )
        )

        mask_results = dict(
            mask_preds=mask_preds,
            mask_feats=mask_feats,
            iou_predictions=iou_predictions,
            mask_preds_bo=mask_preds_bo,
        )
        return mask_results

    def mask_loss(
        self,
        x: Tuple[Tensor],
        sampling_results: List[SamplingResult],
        bbox_feats: Tensor,
        batch_gt_instances: InstanceList,
        image_embeddings=None,
        image_positional_embeddings=None,
    ) -> dict:
        if not self.share_roi_extractor:
            pos_rois = bbox2roi([res.pos_priors for res in sampling_results])
            if len(pos_rois) == 0:
                return dict(
                    loss_mask=dict(loss_mask=0 * x[0].sum()),
                    loss_mask_bo=dict(loss_mask_bo=0 * x[0].sum()),
                )

            mask_results = self._mask_forward(
                x,
                pos_rois,
                image_embeddings=image_embeddings,
                image_positional_embeddings=image_positional_embeddings,
            )

        else:
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                pos_inds.append(
                    torch.ones(
                        res.pos_priors.shape[0], device=device, dtype=torch.uint8
                    )
                )
                pos_inds.append(
                    torch.zeros(
                        res.neg_priors.shape[0], device=device, dtype=torch.uint8
                    )
                )
            pos_inds = torch.cat(pos_inds)

            mask_results = self._mask_forward(
                x, pos_inds=pos_inds, bbox_feats=bbox_feats
            )

        mask_loss_and_target = self.mask_head.loss_and_target(
            mask_preds=mask_results["mask_preds"],
            mask_bo_preds=mask_results["mask_preds_bo"],
            sampling_results=sampling_results,
            batch_gt_instances=batch_gt_instances,
            rcnn_train_cfg=self.train_cfg,
        )

        mask_results.update(
            loss_mask=mask_loss_and_target["loss_mask"],
            loss_mask_bo=mask_loss_and_target["loss_mask_bo"],
        )
        return mask_results

    def loss(
        self,
        x: Tuple[Tensor],
        rpn_results_list: InstanceList,
        batch_data_samples: List[DetDataSample],
        # extra inputs
        image_embeddings=None,
        image_positional_embeddings=None,
    ) -> dict:
        assert len(rpn_results_list) == len(batch_data_samples)
        outputs = unpack_gt_instances(batch_data_samples)
        # batch_gt_instances: List [InstanceData(labels, masks, bboxes, bo_masks)]
        # batch_img_metas: List [dict{'batch_input_shape', 'flip', 'img_path', 'ori_shape', 'pad_shape'}]
        batch_gt_instances, batch_gt_instances_ignore, batch_img_metas = outputs

        if hasattr(self, "extra_pe"):
            bs, _, h, w = x[0].shape
            mask_pe = torch.zeros((bs, h, w), device=x[0].device, dtype=torch.bool)
            img_feats_pe = self.extra_pe(mask_pe)
            outputs = []
            for i in range(len(x)):
                output = x[i] + F.interpolate(
                    img_feats_pe,
                    size=x[i].shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                outputs.append(output)
            x = tuple(outputs)

        # assign gts and sample proposals
        num_imgs = len(batch_data_samples)
        sampling_results = []
        for i in range(num_imgs):
            # rename rpn_results.bboxes to rpn_results.priors
            rpn_results = rpn_results_list[i]
            rpn_results.priors = rpn_results.pop("bboxes")

            assign_result = self.bbox_assigner.assign(
                rpn_results, batch_gt_instances[i], batch_gt_instances_ignore[i]
            )
            sampling_result = self.bbox_sampler.sample(
                assign_result,
                rpn_results,
                batch_gt_instances[i],
                feats=[lvl_feat[i][None] for lvl_feat in x],
            )
            sampling_results.append(sampling_result)

        losses = dict()
        # bbox head loss
        if self.with_bbox:
            bbox_results = self.bbox_loss(x, sampling_results)
            losses.update(bbox_results["loss_bbox"])

        # mask head forward and loss
        if self.with_mask:
            mask_results = self.mask_loss(
                x,
                sampling_results,
                bbox_results["bbox_feats"],
                batch_gt_instances,
                image_embeddings=image_embeddings,
                image_positional_embeddings=image_positional_embeddings,
            )
            losses.update(mask_results["loss_mask"])
            losses.update(mask_results["loss_mask_bo"])

        return losses


@MODELS.register_module()
class OcPrompterAnchorMaskHead(RSPrompterAnchorMaskHead):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def loss_and_target(
        self,
        mask_preds: Tensor,
        mask_bo_preds: Tensor,
        sampling_results: List[SamplingResult],
        batch_gt_instances: InstanceList,
        rcnn_train_cfg: ConfigDict,
    ) -> dict:
        mask_targets, mask_bo_targets = self.get_targets(
            sampling_results=sampling_results,
            batch_gt_instances=batch_gt_instances,
            rcnn_train_cfg=rcnn_train_cfg,
        )

        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        # resize to mask_targets size
        mask_preds = F.interpolate(
            mask_preds,
            size=mask_targets.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        mask_bo_preds = F.interpolate(
            mask_bo_preds,
            size=mask_bo_targets.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        loss = dict()
        loss_bo = dict()
        if mask_preds.size(0) == 0:
            loss_mask = mask_preds.sum()
        else:
            if self.class_agnostic:
                loss_mask = self.loss_mask(
                    mask_preds, mask_targets, torch.zeros_like(pos_labels)
                )
            else:
                loss_mask = self.loss_mask(mask_preds, mask_targets, pos_labels)

        if mask_bo_preds.size(0) == 0:
            loss_mask_bo = mask_bo_preds.sum()
        else:
            if mask_bo_targets.size(0) == 0:
                loss_mask_bo = mask_bo_preds.sum()
            else:
                if self.class_agnostic:
                    loss_mask_bo = self.loss_mask(
                        mask_bo_preds, mask_bo_targets, torch.zeros_like(pos_labels)
                    )
                else:
                    loss_mask_bo = self.loss_mask(
                        mask_bo_preds, mask_bo_targets, pos_labels
                    )

        loss["loss_mask"] = loss_mask
        loss_bo["loss_mask_bo"] = 0.25 * loss_mask_bo
        return dict(
            loss_mask=loss,
            mask_targets=mask_targets,
            loss_mask_bo=loss_bo,
            mask_bo_targets=mask_bo_targets,
        )

    def get_targets(
        self,
        sampling_results: List[SamplingResult],
        batch_gt_instances: InstanceList,
        rcnn_train_cfg: ConfigDict,
    ) -> Tensor:
        pos_proposals = [res.pos_priors for res in sampling_results]
        pos_assigned_gt_inds = [res.pos_assigned_gt_inds for res in sampling_results]

        gt_masks = [res.masks for res in batch_gt_instances]
        gt_bo_masks = [res.bo_masks for res in batch_gt_instances]
        mask_targets_list = []
        mask_bo_targets_list = []
        mask_size = rcnn_train_cfg.mask_size
        device = pos_proposals[0].device

        for pos_gt_inds, gt_mask, gt_bo_mask in zip(
            pos_assigned_gt_inds, gt_masks, gt_bo_masks
        ):
            if len(pos_gt_inds) == 0:
                mask_targets = torch.zeros(
                    (0,) + mask_size, device=device, dtype=torch.float32
                )
                mask_bo_targets = torch.zeros(
                    (0,) + mask_size, device=device, dtype=torch.float32
                )
            else:
                mask_targets = gt_mask[pos_gt_inds.cpu()].to_tensor(
                    dtype=torch.float32, device=device
                )
                mask_bo_targets = gt_bo_mask[pos_gt_inds.cpu()].to_tensor(
                    dtype=torch.float32, device=device
                )

            mask_targets_list.append(mask_targets)
            mask_bo_targets_list.append(mask_bo_targets)
        mask_targets = torch.cat(mask_targets_list)
        mask_bo_targets = torch.cat(mask_bo_targets_list)
        return mask_targets, mask_bo_targets

    def forward(
        self,
        x,
        image_embeddings,
        image_positional_embeddings,
        roi_img_ids=None,
        if_bo=False,
        res_img_embeddings=None,
        res_point_embeddings=None,
    ):
        img_bs = image_embeddings.shape[0]
        roi_bs = x.shape[0]
        image_embedding_size = image_embeddings.shape[-2:]

        point_embedings = self.point_emb(x)
        point_embedings = einops.rearrange(
            point_embedings, "b (n c) -> b n c", n=self.per_pointset_point
        )
        if self.with_sincos:
            point_embedings = (
                torch.sin(point_embedings[..., ::2]) + point_embedings[..., 1::2]
            )

        # (B * N_set), N_point, C
        sparse_embeddings = point_embedings.unsqueeze(1)
        num_roi_per_image = torch.bincount(roi_img_ids.long())
        # deal with the case that there is no roi in an image
        num_roi_per_image = torch.cat(
            [
                num_roi_per_image,
                torch.zeros(
                    img_bs - len(num_roi_per_image),
                    device=num_roi_per_image.device,
                    dtype=num_roi_per_image.dtype,
                ),
            ]
        )

        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            roi_bs, -1, image_embedding_size[0], image_embedding_size[1]
        )
        # get image embeddings with num_roi_per_image
        image_embeddings = image_embeddings.repeat_interleave(num_roi_per_image, dim=0)
        image_positional_embeddings = image_positional_embeddings.repeat_interleave(
            num_roi_per_image, dim=0
        )

        low_res_masks, iou_predictions, _, img_embeddings, point_embeddings = (
            self.mask_decoder(
                image_embeddings=image_embeddings,
                image_positional_embeddings=image_positional_embeddings,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=self.multimask_output,
                attention_similarity=self.attention_similarity,
                target_embedding=self.target_embedding,
                output_attentions=self.output_attentions,
                if_bo=if_bo,
                res_img_embeddings=res_img_embeddings,
                res_point_embeddings=res_point_embeddings,
            )
        )
        h, w = low_res_masks.shape[-2:]
        low_res_masks = low_res_masks.reshape(roi_bs, -1, h, w)
        iou_predictions = iou_predictions.reshape(roi_bs, -1)
        return low_res_masks, iou_predictions, img_embeddings, point_embeddings


class OCSamMaskDecoder(SamMaskDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_positional_embeddings: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        output_attentions: Optional[bool] = None,
        attention_similarity: torch.Tensor = None,
        target_embedding: torch.Tensor = None,
        if_bo: bool = False,
        res_img_embeddings: torch.Tensor = None,
        res_point_embeddings: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_channels, height, width = image_embeddings.shape
        point_batch_size = sparse_prompt_embeddings.shape[1]
        # Concatenate output tokens
        output_tokens = torch.cat(
            [self.iou_token.weight, self.mask_tokens.weight], dim=0
        )
        output_tokens = output_tokens.repeat(batch_size, point_batch_size, 1, 1)

        # tokens shape [n, 1, 10, 256]; point_embeddings shape [n, 1, 10, 256]
        if sparse_prompt_embeddings.sum().item() != 0:
            tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=2)
        else:
            tokens = output_tokens
        point_embeddings = tokens.to(self.iou_token.weight.dtype)

        image_embeddings = image_embeddings + dense_prompt_embeddings
        image_embeddings = image_embeddings.repeat_interleave(point_batch_size, 0)
        image_positional_embeddings = image_positional_embeddings.repeat_interleave(
            point_batch_size, 0
        )

        if not if_bo:
            point_embeddings = point_embeddings + res_point_embeddings
            image_embeddings = image_embeddings + res_img_embeddings

        point_embedding, image_embeddings, attentions = self.transformer(
            point_embeddings=point_embeddings,
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
            attention_similarity=attention_similarity,
            target_embedding=target_embedding,
            output_attentions=output_attentions,
        )

        # iou_token_out shape [n, 1, 256]; mask_tokens_out shape [n, 1, 4, 256]
        iou_token_out = point_embedding[:, :, 0, :]
        mask_tokens_out = point_embedding[:, :, 1 : (1 + self.num_mask_tokens), :]

        # image_embeddings shape [n, 256, 32, 32]
        # Upscale mask embeddings and predict masks using the mask tokens
        image_embeddings = image_embeddings.transpose(2, 3).reshape(
            batch_size * point_batch_size, num_channels, height, width
        )

        point_embeddings_return = point_embedding.clone()
        image_embeddings_return = image_embeddings.clone()

        # upscaled_embedding shape [n, 32, 128, 128]
        upscaled_embedding = self.upscale_conv1(image_embeddings)
        upscaled_embedding = self.activation(
            self.upscale_layer_norm(upscaled_embedding)
        )
        upscaled_embedding = self.activation(self.upscale_conv2(upscaled_embedding))

        hyper_in_list = []
        # hyper_in shape [n, 1, 4, 256]
        for i in range(self.num_mask_tokens):
            current_mlp = self.output_hypernetworks_mlps[i]
            hyper_in_list += [current_mlp(mask_tokens_out[:, :, i, :])]
        hyper_in = torch.stack(hyper_in_list, dim=2)
        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        # upscaled_embedding shape [n, 1, 32, 16384]; masks shape [n, 1, 4, 128, 128]; iou_pred shape [n, 1, 4]
        _, num_channels, height, width = upscaled_embedding.shape
        upscaled_embedding = upscaled_embedding.reshape(
            batch_size, point_batch_size, num_channels, height * width
        )
        masks = (hyper_in @ upscaled_embedding).reshape(
            batch_size, point_batch_size, -1, height, width
        )

        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)

        # masks shape [n, 1, 1, 128, 128]; iou_pred shape [n, 1, 1]
        masks = masks[:, :, mask_slice, :, :]
        iou_pred = iou_pred[:, :, mask_slice]

        outputs = (masks, iou_pred)

        if output_attentions:
            outputs = outputs + (attentions,)
        else:
            outputs = outputs + (None,)

        outputs = outputs + (image_embeddings_return,)
        outputs = outputs + (point_embeddings_return,)
        return outputs


@MODELS.register_module()
class OcRSSamMaskDecoder(RSSamMaskDecoder):
    def __init__(
        self,
        hf_pretrain_name,
        extra_config=None,
        init_cfg=None,
    ):
        BaseModule.__init__(self, init_cfg=init_cfg)
        sam_config = SamConfig.from_pretrained(hf_pretrain_name).mask_decoder_config
        if extra_config is not None:
            sam_config.update(extra_config)
        self.mask_decoder = OCSamMaskDecoder(sam_config)
        self.mask_decoder_bo = OCSamMaskDecoder(sam_config)

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_positional_embeddings: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        output_attentions: Optional[bool] = None,
        attention_similarity: torch.Tensor = None,
        target_embedding: torch.Tensor = None,
        if_bo=bool,
        res_img_embeddings: torch.Tensor = None,
        res_point_embeddings: torch.Tensor = None,
    ):
        if if_bo:
            return self.mask_decoder_bo(
                image_embeddings,
                image_positional_embeddings,
                sparse_prompt_embeddings,
                dense_prompt_embeddings,
                multimask_output,
                output_attentions,
                attention_similarity,
                target_embedding,
                if_bo,
                None,
                None,
            )
        else:
            return self.mask_decoder(
                image_embeddings,
                image_positional_embeddings,
                sparse_prompt_embeddings,
                dense_prompt_embeddings,
                multimask_output,
                output_attentions,
                attention_similarity,
                target_embedding,
                if_bo,
                res_img_embeddings,
                res_point_embeddings,
            )
