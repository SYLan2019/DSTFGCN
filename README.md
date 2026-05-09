# DSTFGCN

DSTFGCN: A Dynamic Spatial-temporal Fusion Graph Convolution Network for Traffic Flow Forecasting

This repository contains the implementation of DSTFGCN, a deep learning model for traffic flow forecasting.

## Environment

To run this code, please ensure you have the following dependencies installed:

* Python 3.9.7
* Torch 1.10.2
* NumPy 1.23.5

## Dataset

Step 1: Download the dataset from [Baidu Yun](https://pan.baidu.com/s/17Gx55aVhqMpAePet4g0tkg?pwd=gfwy) (Access Code: gfwy).

Step 2: Put the four .npz files into their respective folders.

Step 3: Run pre\_process\_data.py to obtain input for the model.

## Train command

```bash
python train.py --config_path='config/PeMS08.json' --device='cuda:0'
```

<button class="copy-btn" onclick="navigator.clipboard.writeText('python train.py --config_path=\\\\'config/PeMS08.json\\\\' --device=\\\\'cuda:0\\\\'')"></button>

Training other datasets only requires modifying the corresponding variables.

## Cite

If you use this repository or would like to refer the paper, please use the following BibTeX entry
```
@article{DSTFGCN,
author = {Tianyi Pan and Xinyuan Zhou and Shiyong Lan and Wenwu Wang and Hongyu Yang and Zheng Li and Zhiang Hou and Yao Ren},
title = {DSTFGCN: A dynamic spatial-temporal fusion graph convolution network for traffic flow forecasting},
journal = {Neural Networks},
volume = {201},
pages = {108989},
year = {2026},
issn = {0893-6080},
doi = {https://doi.org/10.1016/j.neunet.2026.108989}}
```


## 

