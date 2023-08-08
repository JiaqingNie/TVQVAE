import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import tqdm

from torchvision.utils import make_grid
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from cifar10 import save_image, denorm
from modules import *

ckpt = "1689868449"
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 4

    mean = (0.491, 0.482, 0.447)
    std = (0.247, 0.243, 0.262)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    test_dat = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)

    loader_test = DataLoader(test_dat, batch_size, shuffle=False)

    model = torch.load(f'./TVQVAE/results/{ckpt}/TVQVAE.pt')
    model.eval()

    with torch.no_grad():
        # randomly choose 10 images from test set and for each image, generate 10 samples, save them and the original image in a grid
            # get 10 images from loader_test
        for j, (x, y) in enumerate(loader_test):
            if j == 10:
                break
            x = x.to(device)
            y = y.to(device)
            z = model.encoder(x, y)
            img1 = make_grid(torch.clamp(denorm(z.cpu(), mean, std), 0., 1.), nrow=4, padding=0, normalize=False,
                                    range=None, scale_each=False, pad_value=0)
            z, _ = model.codebook.straight_through(z)
            img2 = make_grid(torch.clamp(denorm(z.cpu(), mean, std), 0., 1.), nrow=4, padding=0, normalize=False,
                                    range=None, scale_each=False, pad_value=0)
            recons = model.decoder(z, y)
            recons = recons.cpu()
            recons = make_grid(torch.clamp(denorm(recons, mean, std), 0., 1.), nrow=4, padding=0, normalize=False,
                                    range=None, scale_each=False, pad_value=0)
            xs = x.cpu()
            xs = make_grid(torch.clamp(denorm(xs, mean, std), 0., 1.), nrow=4, padding=0, normalize=False,
                                    range=None, scale_each=False, pad_value=0)
            plt.figure(figsize = (8,8))
            save_image(img1, f"./TVQVAE/eval/{ckpt}", f"z_{j}")   
            plt.figure(figsize = (8,8))
            save_image(img2, f"./TVQVAE/eval/{ckpt}", f"latent_{j}") 
            plt.figure(figsize = (8,8))
            save_image(recons, f"./TVQVAE/eval/{ckpt}", f"recons_{j}")
            plt.figure(figsize = (8,8))
            save_image(xs, f"./TVQVAE/eval/{ckpt}", f"xs_{j}")

if __name__ == '__main__':
    main()

