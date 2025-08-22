# DSTFGCN

DSTFGCN: A Dynamic Spatial-temporal Fusion Graph Convolution Network for Traffic Flow Forecasting

## Environment

Python 3.9.7

Torch 2.0.1

NumPy 1.24.0

## Dataset

Step 1: Download the dataset from https://pan.baidu.com/s/1l6IA3PqRDYTsCsLr00n7Lw?pwd=pems.

Step 2: Put the four .npz files into their respective folders.

Step 3: Run pre\_process\_data.py to obtain input for the model.

## Train command

python train.py  --config\_path='config/PeMS08.json' --device='cuda:0'

Training other datasets only requires modifying the corresponding variables.



## 

