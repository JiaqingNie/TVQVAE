# TVQVAE

This project implements a Transformer-based Vector Quantized VAE.

## Dependencies

```xml
- python >= 3.8
- pytorch >= 1.13
- torchvision
- pytorch-cuda=11.7
- pip
- pip:
- timm
- diffusers
- lmdb
- accelerate
- taming-transformers
- einops
- matplotlib
- pytorch_lightning
```

## How to run

For training on CIFAR-10 dataset, run:

```bash
python cifar10.py
```

For training on LSUN dataset, run:

```bash
python lsun.py
```

