# Prohibited Items Segmentation via Occlusion-aware Bilayer Modeling
Official PyTorch implementation of Prohibited Items Segmentation via Occlusion-aware Bilayer Modeling.

This paper is accepted by ICME 2025.

## Get Started
### Dependencies
We test our code using Linux system with Pytorch 2.1.2 and CUDA 11.8.

If necessary, install numpy with version lower than 2.0.0 (we use 1.24.0).
```
conda create -n occ python=3.10
conda activate occ
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install -U openmim
mim install mmcv==2.1.0
conda install mpi4py
pip install -U transformers==4.38.1 wandb==0.16.3 einops pycocotools shapely scipy terminaltables importlib peft==0.8.2 mat4py==0.6.0
```

### Prepare Model Weights
Pre-download the SAM weights. For example, 'SAM-vit-large' can be downloaded using the following script:
```
from transformers import SamModel
import os

hf_pretrain_name = "facebook/sam-vit-large"

cache_dir = f"{os.path.basename(hf_pretrain_name).replace('-', '_')}"
os.makedirs(cache_dir, exist_ok=True)

model = SamModel.from_pretrained(hf_pretrain_name, use_safetensors=False)
model.save_pretrained(cache_dir, safe_serialization=False)
```

After doing so, change the paths in the 'convert_model_weights.py' script accordingly and run it to provide initial weights for Occluder Mask Decoder.

### Prepare Datasets
Download Occ-PIDray dataset from this link:

Download Occ-PIXray dataset from this link:

We also provide script 'convert_occlusion_annotation.py' which generate occlusion annotation for customized COCO format datasets. To generate your own occlusion-annotated datasets, change paths accordingly in the script and run it.

### Modify Config Files
Modify the paths in config files 'configs/occlusion/pid.py' and 'configs/occlusion/pix.py' accordingly.



## Running
All our code are tested and running successfully on both 1 NVIDIA A800 Tensor Core GPU with 80GB VRAM and 8 NVIDIA GeForce RTX 4090 GPUs.

To train our model:
```
# Singlue GPU, OccPIDray
CUDA_VISIBLE_DEVICES=0 bash ./tools/dist_train.sh configs/occlusion/pid.py 1
# Singlue GPU, OccPIXray
CUDA_VISIBLE_DEVICES=0 bash ./tools/dist_train.sh configs/occlusion/pix.py 1


# Multiple GPUs, OccPIDray
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./tools/dist_train.sh configs/occlusion/pid.py 8
# Multiple GPUs, OccPIXray
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./tools/dist_train.sh configs/occlusion/pix.py 8
```

To test our model, change the 'CHECKPOINT_FILE' and 'GPU_NUM' to your own checkpoint path and GPU numbers:
```
# OccPIDray
bash ./tools/dist_test.sh configs/occlusion/pid.py /root/autodl-tmp/Occlusion/OccPIDray.pth 1
# OccPIXray
bash ./tools/dist_test.sh configs/occlusion/pix.py /root/autodl-tmp/Occlusion/OccPIXray.pth 1
```

Additionally, we provide our trained model checkpoints in the following links:
OccPIDray:
OccPIXray:


## Thanks
This code is based on [RSPrompter](https://github.com/KyanChen/RSPrompter/) and [MMDetection](https://github.com/open-mmlab/mmdetection). Many thanks for your code implementation.


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
