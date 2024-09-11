# visual-audio cross modal retrieval task
Code for Paper: Enhancing Semantic Audio-Visual Representation Learning with Supervised Multi-Scale Attention. Our code runs in the Windows 11 environment.
# Installation
## 1. Clone the repository
```bash
$ git clone https://github.com/zhangjiwei-japan/SMSA-VA-CMR.git
```
## 2. Requirements
#### （1） Install python from website ：https://www.python.org/downloads/windows/
#### （2） Our program runs on the GPU. Please install the cuda, cudnn, etc as follows : 
- CUDA Toolkit Archive ：https://developer.nvidia.com/cuda-toolkit-archive
- cuDNN Download | NVIDIA Developer ：https://developer.nvidia.com/login
- PYTORCH : https://pytorch.org/

## 3. Prepare the datasets
### (1) AVE Dataset 
- AVE dataset can be downloaded from https://drive.google.com/open?id=1FjKwe79e0u96vdjIVwfRQ1V6SoDHe7kK.
- Scripts for generating audio and visual features: https://drive.google.com/file/d/1TJL3cIpZsPHGVAdMgyr43u_vlsxcghKY/view?usp=sharing
#### You can also download our prepared [AVE](https://drive.google.com/file/d/14Qdprd8_9cdih3QDN726kJTzaoo9Y8Y-/view?usp=sharing) dataset.
- Please create a folder named 'ave' to place the downloaded dataset and set the dataset base path in the code: `base_dir` = "./datasets/ave/"
- Place the downloaded dataset in the 'ave' file and load a dataset path in the code: `load_path` = `base_dir` + "your downloaded dataset"
### (2) VEGAS Dataset 
- The Raw dataset from: https://arxiv.org/abs/1712.01393.
#### You can also download our prepared [VEGAS](https://drive.google.com/file/d/142VXU9-3P2HcaCWCQVlezRGJguGnHeHD/view?usp=sharing) dataset. 
- Please create a folder named 'veags' to place the downloaded dataset and set the dataset base path in the code (train_model.py and test_model.py): `base_dir` = "./datasets/vegas/"
- Place the downloaded dataset in the 'veags' file and load a dataset path in the code (train_model.py and test_model.py): `load_path` = `base_dir` + "your downloaded dataset" <br />

You can download and use the model files created by training with SMSA on two different datasets for inference.<br />
Training/Inference code and configuration files will be added.
