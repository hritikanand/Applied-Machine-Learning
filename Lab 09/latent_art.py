# latent_art.py
import sys
sys.path.append('./stylegan2-ada-pytorch')  # Add path to stylegan2 repo

import torch
import numpy as np
import PIL.Image
import dnnlib
import legacy
import os

# Create output directory
os.makedirs("generated_outputs", exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
network_pkl = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl'

print('🔄 Loading network...')
with dnnlib.util.open_url(network_pkl) as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device)

# Generate two random latent vectors
z1 = torch.randn([1, G.z_dim]).to(device)
z2 = torch.randn([1, G.z_dim]).to(device)

# Create a series of interpolated images
for i, alpha in enumerate(np.linspace(0, 1, 8)):
    z = (1 - alpha) * z1 + alpha * z2
    img = G(z, None, noise_mode='const')[0]
    img = (img.clamp(-1, 1) + 1) * (255 / 2)
    img = img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    PIL.Image.fromarray(img).save(f"generated_outputs/interp_{i:02d}.png")

print("✅ Latent artwork images saved in 'generated_outputs/'")
