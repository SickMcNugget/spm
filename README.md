# Explicit-Shape-Priors

This repo is the official implementation for: [Learning with Explicit Shape Priors for Medical Image Segmentation](https://arxiv.org/abs/2303.17967)


# Dataset Link
[BraTS 2020: Multimodal Brain Tumor Segmentation Challenge 2020](https://www.med.upenn.edu/cbica/brats2020/data.html)  

[VerSe'19: Large Scale Vertebrae Segmentation Challenge](https://verse2019.grand-challenge.org/)  

[Automated Cardiac Diagnosis Challenge (ACDC)](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html)  

# Preprocess
We follow the z-score normalization strategy in [nnUNet](https://github.com/MIC-DKFZ/nnUNet) to preprocess the BraTS 2020, VerSe'19 and ACDC dataset.

# Requirements
* python 3.7  
* [pytorch 1.8.0](https://pytorch.org/get-started/previous-versions/#v180)
* torchvision 0.9.0
* simpleitk 2.0.2
* monai 0.9.0

# Install
Start by creating a new conda environment and installing pytorch:
```bash
conda create -n spm python=3.7 -y
conda activate spm
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch -y
```
After that, install the rest of the requirements from PyPI:
```bash
pip install -r requirements.txt
```

# Training
If you want to train the model from scratch, run the training script as following.  
`python BraTS_train.py`  
`python VerSe_train.py`  
`python ACDC_train.py`


# Testing
If you want to test the model, run the testing script as following.  
`python BraTS_test.py`  
`python VerSe_test.py`  
`python ACDC_test.py`

# Citation
If you use our code or models in your work or find it is helpful, please cite the corresponding paper:  
```
@article{you2023learning,
  title={Learning with Explicit Shape Priors for Medical Image Segmentation},
  author={You, Xin and He, Junjun and Yang, Jie and Gu, Yun},
  journal={arXiv preprint arXiv:2303.17967},
  year={2023}
}
```
