!pip install matplotlib torch torchvision tqdm numpy ipykernel torch-fidelity pandas

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch_fidelity
import pandas as pd

import os
import time
import pickle
import zipfile
import numpy as np
import urllib.request
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import torchvision.models as models


seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ------------------------------------------------------------
# 0) Hyperparameters / config
# ------------------------------------------------------------
# Define some params
DATASET_PATH =  '/common/home/users/a/annamalaik.2022/cs-424-group-project-friday/task_2_data/'
DEVICE = 'cuda'  # 'cpu' or 'cuda'
OUTPUT_PATH = '/common/home/users/a/annamalaik.2022/cs-424-group-project-friday/work_dirs/Cycle_GAN_02'
CHECKPOINT_SAVE_EVERY = 2
LATENT_DIMS = 100
LR = 0.0002
WORKERS = 4
NETWORK_FEATURES = 128
EPOCH = 200
BATCH_SIZE = 8


# ------------------------------------------------------------
# 1) Define a custom Dataset that returns (cat_img, dog_img)
# ------------------------------------------------------------
# CycleGAN uses *unpaired* image-to-image translation:
# - We do NOT need aligned (apple_i, orange_i) pairs.
# - Instead, during training we sample an apple image from domain A
#   and an orange image from domain B independently.
#
# This dataset class does exactly that by indexing with modulo:
# - apple index cycles through trainA images
# - orange index cycles through trainB images
# and the length is max(len(A), len(B)) so an epoch covers the larger domain.
class Anime2RealDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None, train_or_test='train'):
        self.dir_A = os.path.join(root_dir, 'anime', train_or_test)
        self.dir_B = os.path.join(root_dir, 'real', train_or_test)

        self.transform = transform

        self.images_A = sorted([f for f in os.listdir(self.dir_A) if not f.startswith('.')])
        self.images_B = sorted([f for f in os.listdir(self.dir_B) if not f.startswith('.')])

        self.len_A = len(self.images_A)
        self.len_B = len(self.images_B)

        self.length_dataset = max(self.len_A, self.len_B)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        img_A = self.images_A[index % self.len_A]
        img_B = self.images_B[index % self.len_B]

        path_A = os.path.join(self.dir_A, img_A)
        path_B = os.path.join(self.dir_B, img_B)

        image_A = Image.open(path_A).convert("RGB")
        image_B = Image.open(path_B).convert("RGB")

        if self.transform:
            image_A = self.transform(image_A)
            image_B = self.transform(image_B)

        return image_A, image_B

transform = transforms.Compose([
    transforms.Resize(286),
    transforms.CenterCrop(256),   # more stable for faces
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3)
])


dataset = Anime2RealDataset(DATASET_PATH, transform=transform, train_or_test='train')

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=WORKERS,
    pin_memory=True,
    drop_last=True
)
print("hello task 2")

lambda_gan = 5
lambda_cycle = 10          # slightly stronger structure
lambda_feat = 0.02        # stabilize textures a bit more
lambda_perc = 0        # perceptual loss to stabilize high-level features
lambda_id = 0

# -----------------------------
# MODEL BLOCKS
# -----------------------------
from torch.nn.utils import spectral_norm

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, 128, 3, 2, 1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),

            nn.Conv2d(128, 256, 3, 2, 1),
            nn.InstanceNorm2d(256),
            nn.ReLU(True),

            *[ResidualBlock(256) for _ in range(9)],

            nn.ConvTranspose2d(256, 128, 3, 2, 1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 3, 2, 1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, 7),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.ModuleList([
            nn.Sequential(
                spectral_norm(nn.Conv2d(3, 64, 4, 2, 1)),
                nn.LeakyReLU(0.2)
            ),
            nn.Sequential(
                spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)),
                nn.InstanceNorm2d(128),
                nn.LeakyReLU(0.2)
            ),
            nn.Sequential(
                spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)),
                nn.InstanceNorm2d(256),
                nn.LeakyReLU(0.2)
            ),
            nn.Sequential(
                spectral_norm(nn.Conv2d(256, 512, 4, 1, 1)),
                nn.InstanceNorm2d(512),
                nn.LeakyReLU(0.2)
            ),
            spectral_norm(nn.Conv2d(512, 1, 5, 1, 2))
        ])

    def forward(self, x, return_features=False):
        feats = []
        for layer in self.layers[:-1]:
            x = layer(x)
            feats.append(x)
        out = self.layers[-1](x)

        if return_features:
            return out, feats
        return out

class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.D1 = Discriminator()
        self.D2 = Discriminator()
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1)

    def forward(self, x, return_features=False):
        x_down = self.downsample(x)

        if return_features:
            out1, feat1 = self.D1(x, return_features=True)
            out2, feat2 = self.D2(x_down, return_features=True)
            return [out1, out2], [feat1, feat2]
        else:
            return [self.D1(x), self.D2(x_down)]

class ReplayBuffer:
    def __init__(self, max_size=50):
        self.data = []
        self.max_size = max_size

    def push_and_pop(self, batch):
        result = []
        for img in batch:
            img = img.unsqueeze(0)
            if len(self.data) < self.max_size:
                self.data.append(img)
                result.append(img)
            else:
                if random.random() > 0.5:
                    idx = random.randint(0, self.max_size - 1)
                    result.append(self.data[idx])
                    self.data[idx] = img
                else:
                    result.append(img)
        return torch.cat(result)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)

anime_G = Generator().to(DEVICE)   # cat → dog
real_G = Generator().to(DEVICE)   # dog → cat

anime_D = MultiScaleDiscriminator().to(DEVICE)
real_D  = MultiScaleDiscriminator().to(DEVICE) 

criterion_GAN = nn.MSELoss()
criterion_cycle = nn.L1Loss()
criterion_identity = nn.L1Loss()

def gan_loss(preds, target):
    loss = 0
    for p in preds:
        loss += criterion_GAN(p, target)
    return loss

def feature_matching_loss_multi(fake, real, D, lambda_var=0.5):
    _, fake_feats_all = D(fake, return_features=True)
    _, real_feats_all = D(real, return_features=True)

    loss = 0.0

    for fake_feats, real_feats in zip(fake_feats_all, real_feats_all):
        for f_fake, f_real in zip(fake_feats, real_feats):
            f_real = f_real.detach()

            B, C, H, W = f_fake.shape
            f_fake = f_fake.view(B, C, -1)
            f_real = f_real.view(B, C, -1)

            mean_fake = f_fake.mean(dim=(0,2))
            mean_real = f_real.mean(dim=(0,2))

            var_fake = f_fake.var(dim=(0,2), unbiased=False)
            var_real = f_real.var(dim=(0,2), unbiased=False)

            loss += (mean_fake - mean_real).abs().mean()
            loss += lambda_var * (var_fake - var_real).abs().mean()

    return loss

opt_G = torch.optim.Adam(
    list(anime_G.parameters()) + list(real_G.parameters()),
    lr=0.0002, betas=(0.5, 0.999)
)

opt_D_A = torch.optim.Adam(anime_D.parameters(), lr=0.0003, betas=(0.5, 0.999))
opt_D_B = torch.optim.Adam(real_D.parameters(), lr=0.0003, betas=(0.5, 0.999))

decay_start_epoch = 120

def lambda_rule(epoch):
    return 1.0 - max(0, epoch - decay_start_epoch) / (EPOCH - decay_start_epoch)

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(opt_G, lr_lambda=lambda_rule)
lr_scheduler_anime_D = torch.optim.lr_scheduler.LambdaLR(opt_D_A, lr_lambda=lambda_rule)
lr_scheduler_real_D  = torch.optim.lr_scheduler.LambdaLR(opt_D_B, lr_lambda=lambda_rule)

fake_anime_buffer = ReplayBuffer()
fake_real_buffer  = ReplayBuffer()

print(DEVICE)


chk_pt_path = "/common/home/users/a/annamalaik.2022/cs-424-group-project-friday/work_dirs/Cycle_GAN_02/checkpoints/ckpt_14.pth"
if os.path.exists(chk_pt_path):
    ckpt = torch.load("/common/home/users/a/annamalaik.2022/cs-424-group-project-friday/work_dirs/Cycle_GAN_02/checkpoints/ckpt_14.pth", map_location=DEVICE)

    anime_G.load_state_dict(ckpt['anime_G'])
    real_G.load_state_dict(ckpt['real_G'])
    anime_D.load_state_dict(ckpt['anime_D'])
    real_D.load_state_dict(ckpt['real_D'])

    opt_G.load_state_dict(ckpt['optimizer_G'])
    opt_D_A.load_state_dict(ckpt['optimizer_anime_D'])
    opt_D_B.load_state_dict(ckpt['optimizer_real_D'])

    lr_scheduler_G.load_state_dict(ckpt['lr_scheduler_G'])
    lr_scheduler_anime_D.load_state_dict(ckpt['lr_scheduler_anime_D'])
    lr_scheduler_real_D.load_state_dict(ckpt['lr_scheduler_real_D'])

    start_epoch = ckpt['epoch'] + 1
else:
    start_epoch = 0
print(start_epoch)

# -----------------------------
# TRAIN LOOP
# -----------------------------

print("Starting training...")

for epoch in range(start_epoch, EPOCH):
    for real_anime, real_face in dataloader:
        real_anime = real_anime.to(DEVICE)
        real_face = real_face.to(DEVICE)

        # ------------------ GENERATOR ------------------
        opt_G.zero_grad()

        fake_face = anime_G(real_anime)
        fake_anime = real_G(real_face)

        preds_real = real_D(fake_face)
        loss_GAN_A = sum(criterion_GAN(p, torch.ones_like(p)) for p in preds_real)

        preds_anime = anime_D(fake_anime)
        loss_GAN_B = sum(criterion_GAN(p, torch.ones_like(p)) for p in preds_anime)

        rec_anime = real_G(fake_face)
        rec_face = anime_G(fake_anime)

        loss_cycle = criterion_cycle(rec_anime, real_anime) + \
                     criterion_cycle(rec_face, real_face)

        loss_id = criterion_identity(real_G(real_anime), real_anime) + \
                  criterion_identity(anime_G(real_face), real_face)

        loss_feat = feature_matching_loss_multi(fake_face, real_face, real_D) + \
                    feature_matching_loss_multi(fake_anime, real_anime, anime_D)

        loss_G = (lambda_gan * (loss_GAN_A + loss_GAN_B)
                  + lambda_cycle * loss_cycle
                  + lambda_id * loss_id
                  + lambda_feat * loss_feat
                  )

        loss_G.backward()
        opt_G.step()

        # -------- Discriminator anime --------
        opt_D_A.zero_grad()

        pred_real = anime_D(real_anime)
        loss_real = sum(criterion_GAN(p, torch.ones_like(p)) for p in pred_real)

        fake_buf = fake_anime_buffer.push_and_pop(fake_anime.detach())
        pred_fake = anime_D(fake_buf)
        loss_fake = sum(criterion_GAN(p, torch.zeros_like(p)) for p in pred_fake)

        loss_D_A = (loss_real + loss_fake) * 0.5
        loss_D_A.backward()
        opt_D_A.step()

        # -------- Discriminator real --------
        opt_D_B.zero_grad()

        pred_real = real_D(real_face)  # list of outputs
        loss_real = sum(
            criterion_GAN(p, torch.ones_like(p)) for p in pred_real
        )

        fake_buf = fake_real_buffer.push_and_pop(fake_face.detach())
        pred_fake = real_D(fake_buf)  # list of outputs
        loss_fake = sum(
            criterion_GAN(p, torch.zeros_like(p)) for p in pred_fake
        )

        loss_D_B = (loss_real + loss_fake) * 0.5
        loss_D_B.backward()
        opt_D_B.step()

    print(f"Epoch [{epoch+1}/{EPOCH}] | G: {loss_G.item():.4f}")

    lr_scheduler_G.step()
    lr_scheduler_anime_D.step()
    lr_scheduler_real_D.step()

    # -------------------------
    # Optional: Save sample images every 10 epochs
    # -------------------------
    if epoch % 10 == 0:
        with torch.no_grad():
            sample_anime = real_anime[:2]
            sample_face = real_face[:2]

            fake_face_vis = anime_G(sample_anime)
            fake_anime_vis = real_G(sample_face)
    
            from torchvision.utils import save_image
            save_image((fake_anime_vis * 0.5 + 0.5), f"{OUTPUT_PATH}/epoch_{epoch}_anime.png")
            save_image((fake_face_vis * 0.5 + 0.5), f"{OUTPUT_PATH}/epoch_{epoch}_face.png")

    if (epoch + 1) % CHECKPOINT_SAVE_EVERY == 0:
        torch.save({
            'anime_G': anime_G.state_dict(),
            'real_G': real_G.state_dict(),
            'anime_D': anime_D.state_dict(),
            'real_D': real_D.state_dict(),
            'optimizer_G': opt_G.state_dict(),
            'optimizer_anime_D': opt_D_A.state_dict(),
            'optimizer_real_D': opt_D_B.state_dict(),
            'lr_scheduler_G': lr_scheduler_G.state_dict(),
            'lr_scheduler_anime_D': lr_scheduler_anime_D.state_dict(),
            'lr_scheduler_real_D': lr_scheduler_real_D.state_dict(),
            'epoch': epoch
        }, os.path.join(OUTPUT_PATH, 'checkpoints', f'ckpt_{epoch+1}.pth'))
