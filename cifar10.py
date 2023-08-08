# %%
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

from time import time
from modules import *
from TVQVAE import TVQVAE


# %%
def denorm(img_tensors, mean, std):
    # denormalize image tensors with mean and std of training dataset for all channels
    img_tensors = img_tensors.permute(1, 0, 2, 3)
    for t, m, s in zip(img_tensors, mean, std):
        t.mul_(s).add_(m)
    img_tensors = img_tensors.permute(1, 0, 2, 3)
    return img_tensors
    
    
def save_image(img, path, name):
    npimg = img.cpu().numpy()
    plt.axis('off')
    plt.imshow(np.transpose(npimg, (1,2,0)))
    # save image
    if not os.path.exists(path):
        os.mkdir(path)
    plt.savefig(f'{path}/{name}.png')

# %%
data_path = './data'
batch_size = 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mean = (0.5, 0.5, 0.5)
std = (0.5, 0.5, 0.5)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

train_dat = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
test_dat = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

loader_train = DataLoader(train_dat, batch_size, shuffle=True)
loader_test = DataLoader(test_dat, batch_size, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# %%
latent_dim = (8,8,3)
model = TVQVAE(latent_dim=latent_dim, image_size=32, patch_size=2, in_channels=3, hidden_size=768, depth=12, num_heads=6, mlp_ratio=6.0, num_classes=10, dropout_prob=0.1)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# %%
def train_VAE(model, device, loader_train, optimizer, num_epochs, latent_dim, beta_warm_up_period=1):
    assert isinstance(latent_dim, tuple) and len(latent_dim) == 3, "latent_dim must be a tuple of length 3"
    timestamp = int(time())
    
    model.eval()
    with torch.no_grad():
        data, y = next(iter(loader_test))
        data = data.to(device)
        y = y.to(device)
        recon_x, _, _ = model(data, y)
        recon_x = recon_x.cpu()
        recons = make_grid(torch.clamp(denorm(recon_x, mean, std), 0., 1.), nrow=2, padding=0, normalize=False,
                                range=None, scale_each=False, pad_value=0)
        plt.figure(figsize = (8,8))
        save_image(recons, f'./TVQVAE/results/{timestamp}', 0)
    
    train_losses = []
    test_losses = []
    for epoch in range(num_epochs):  
        model.train()
        train_loss = 0
        train_mse_loss = 0
        train_KL_div = 0
        
        beta = 1
        
        with tqdm.tqdm(loader_train, unit="batch") as tepoch: 
            for batch_idx, (data, y) in enumerate(tepoch):   
                data = data.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                recon_x, z_e_x, z_q_x = model(data, y)
                loss_recons = F.mse_loss(recon_x, data)
                loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
                loss_commit = F.mse_loss(z_e_x, z_q_x.detach())
                loss = loss_recons +loss_vq + loss_commit * beta
                loss.backward()
                optimizer.step()
                
                tepoch.set_description(f"Epoch {epoch}")
                tepoch.set_postfix(loss=loss.item()/len(data), loss_recons=loss_recons.item()/len(data), loss_vq=loss_vq.item()/len(data), loss_commit=loss_commit.item()/len(data))
    
    
            
            model.eval()
            with torch.no_grad():
                data, y = next(iter(loader_test))
                data = data.to(device)
                y = y.to(device)
                recon_x,  _, _ = model(data, y)
                recon_x = recon_x.cpu()
                recons = make_grid(torch.clamp(denorm(recon_x, mean, std), 0., 1.), nrow=4, padding=0, normalize=False,
                                        range=None, scale_each=False, pad_value=0)
                plt.figure(figsize = (8,8))
                save_image(recons, f'./TVQVAE/results/{timestamp}', epoch+1)

        # save the model
        if epoch == num_epochs - 1:
            with torch.no_grad():
                torch.save(model, f'./TVQVAE/results/{timestamp}/TVQVAE.pt')
    return train_losses


# %%
if __name__ == '__main__':
    train_losses = train_VAE(model, device, loader_train, optimizer, 8, latent_dim, beta_warm_up_period=10)
