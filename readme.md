# Prohibited Items Segmentation via Occlusion-aware Bilayer Modeling (ICME 2025)
Official PyTorch Implementation of Prohibited Items Segmentation via Occlusion-aware Bilayer Modeling.

## Get Started
### Dependencies
Code is tested on Linux with Pytorch 2.1.2 and CUDA 11.8.

If necessary, install numpy<2.0.0 (we use 1.24.0).
```
conda create -n occ python=3.10
conda activate occ
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install -U openmim
mim install mmcv==2.1.0
conda install mpi4py
pip install -U transformers==4.38.1 wandb==0.16.3 einops pycocotools shapely scipy terminaltables importlib peft==0.8.2 mat4py==0.6.0
conda install importlib-metadata
```

### Prepare Model Weights
Pre-download the SAM weights (e.g., SAM-vit-large) and update the paths in 'convert_model_weights.py'. Weights can be downloaded using the following script:
```
from transformers import SamModel
import os

hf_pretrain_name = "facebook/sam-vit-large"

cache_dir = f"{os.path.basename(hf_pretrain_name).replace('-', '_')}"
os.makedirs(cache_dir, exist_ok=True)

model = SamModel.from_pretrained(hf_pretrain_name, use_safetensors=False)
model.save_pretrained(cache_dir, safe_serialization=False)
```

After doing so, run the 'convert_model_weights.py' script to generate initial weights for the Occluder Mask Decoder.

### Prepare Datasets
Download datasets from the following links, notice that PIXray has been transformed to COCO-format:

[PIDray-A](https://pan.baidu.com/s/1cR0ykp6RAs6lD_ogFxqnjg?pwd=rkz5)

[PIXray-A](https://pan.baidu.com/s/1kAtNeceCtTBc1JnFfcbaSg?pwd=vr2w)

#### Custom Dataset Support
Use 'convert_occlusion_annotation.py' to generate occlusion annotations for COCO-format datasets. To generate your own occlusion-annotated datasets, modify paths in the script before execution.


### Modify Config Files
Update the paths in the configuration files 'configs/occlusion/pid.py' and 'configs/occlusion/pix.py' according to your setup.


## Train & Test
Our code has been successfully tested on:

* 1 × NVIDIA A800 Tensor Core GPU

* 8 × NVIDIA GeForce RTX 4090 GPUs


To train our model:
```
# Single GPU, PIDray-A
CUDA_VISIBLE_DEVICES=0 bash ./tools/dist_train.sh configs/occlusion/pid.py 1
# Single GPU, PIXray-A
CUDA_VISIBLE_DEVICES=0 bash ./tools/dist_train.sh configs/occlusion/pix.py 1

# Multiple GPUs, PIDray-A
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./tools/dist_train.sh configs/occlusion/pid.py 8
# Multiple GPUs, PIXray-A
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./tools/dist_train.sh configs/occlusion/pix.py 8
```

To test our model, set the 'CHECKPOINT_FILE' to your checkpoint path, and configure 'GPU_NUM' to match your GPU setup:
```
# PIDray-A
bash ./tools/dist_test.sh configs/occlusion/pid.py {CHECKPOINT_FILE} {GPU_NUM}
# PIXray-A
bash ./tools/dist_test.sh configs/occlusion/pix.py {CHECKPOINT_FILE} {GPU_NUM}
```

Additionally, we provide model checkpoints in the following link: [Checkpoints](https://pan.baidu.com/s/1KvoAB1V0hB6d7RALb-_FLA?pwd=fcjs)


## Acknowledgements
This implementation builds upon:

* [RSPrompter](https://github.com/KyanChen/RSPrompter/)

* [MMDetection](https://github.com/open-mmlab/mmdetection)

We sincerely appreciate their contributions.

## Cite this paper
```
@inproceedings{ren2025prohibited,
  title={Prohibited Items Segmentation via Occlusion-aware Bilayer Modeling},
  author={Ren, Yunhan and Li, Ruihuang and Liu, Lingbo and Chen,  Changwen},
  booktitle={2025 IEEE International Conference on Multimedia and Expo (ICME)},
  pages={1--6},
  year={2025},
  organization={IEEE}
}
```
