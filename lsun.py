# %%
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import LSUN
from torchvision import datasets
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Subset
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

class LSUNWithLabels:
    def __init__(self, lsun_classes, root_dir, transform=None, num_samples=50000):
        self.datasets = []
        
        for idx, cls in enumerate(lsun_classes):
            full_dataset = LSUN(root=root_dir, classes=[cls], transform=transform)
            
            def wrapped_getitem(index, dataset=full_dataset):
                data, _ = dataset[index]
                return data
            
            indices = torch.randperm(len(full_dataset))[:num_samples]
            subset = [(wrapped_getitem(index), idx) for index in indices]
            self.datasets.append(subset)
            
        self.lengths = [len(subset) for subset in self.datasets]
        self.total_length = sum(self.lengths)

    def __getitem__(self, index):
        for i, subset in enumerate(self.datasets):
            if index < self.lengths[i]:
                return subset[index]
            index -= self.lengths[i]
        raise IndexError

    def __len__(self):
        return self.total_length
# %%
data_path = './data/lsun'
batch_size = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
img_size = 64
num_samples = 50000
mean = (0.5, 0.5, 0.5)
std = (0.5, 0.5, 0.5)
transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


lsun_classes = ['bedroom_train', 'tower_train', 'bridge_train']

print(f"Loading LSUN dataset of classes: {', '.join(lsun_classes)}...")

dataset = LSUNWithLabels(lsun_classes, root_dir=data_path, transform=transform, num_samples=num_samples)
loader_train = DataLoader(dataset, batch_size, shuffle=True)

print("Finished loading LSUN dataset.")

# %%
latent_dim = (16,16,3)
model = TVQVAE(latent_dim=latent_dim, image_size=img_size, patch_size=2, in_channels=3, hidden_size=768, depth=12, num_heads=6, mlp_ratio=6.0, num_classes=3, dropout_prob=0.1)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# %%
def train_VAE(model, device, loader_train, optimizer, num_epochs, latent_dim, beta_warm_up_period=1):
    assert isinstance(latent_dim, tuple) and len(latent_dim) == 3, "latent_dim must be a tuple of length 3"
    timestamp = int(time())
    
    model.eval()
    with torch.no_grad():
        data, y = next(iter(loader_train))
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
                data, y = next(iter(loader_train))
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
    print("Starting training...")
    train_losses = train_VAE(model, device, loader_train, optimizer, 8, latent_dim, beta_warm_up_period=10)
