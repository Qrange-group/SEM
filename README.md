# SEM: Switchable Excitation Module for Self-attention Mechanism
[![996.ICU](https://img.shields.io/badge/link-996.icu-red.svg)](https://996.icu) 
![GitHub](https://img.shields.io/github/license/gbup-group/DIANet.svg)
![GitHub](https://img.shields.io/badge/Qrange%20-group-orange)

This repository is the implementation of "SEM: Switchable Excitation Module for Self-attention Mechanism" [[paper]](https://arxiv.org/abs/?)  on CIFAR-100 and CIFAR-10 datasets. Our paper has been accepted for presentation at ???. You can also check with the [??? proceeding version](???).

## Introduction

SPENet is a self-attention module, which can automatically decide to select and integrate attention operators to compute attention maps.  

<p align="center">
  <img src="https://github.com/Qrange-group/SEM/blob/main/images/arch.png">
</p>

## Requirement
Python and [PyTorch](http://pytorch.org/).
  ```
  pip install -r requirements.txt
  ```
## Usage
  ```
CUDA_VISIBLE_DEVICES=0 python run.py --dataset cifar100 --block-name bottleneck --depth 164 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4
  ```

## Results
|                 |  Dataset  | original |  SEM  |
|:---------------:|:------:|:--------:|:------:|
|    ResNet164    |CIFAR10 |   93.39  |  94.95 |
|    ResNet164    |CIFAR100|   74.30  |  76.76 |



## Citing SEM

```
???
```

## Acknowledgments
Many thanks to [bearpaw](https://github.com/bearpaw) for his simple and clean [Pytorch framework](https://github.com/bearpaw/pytorch-classification) for image classification task.
