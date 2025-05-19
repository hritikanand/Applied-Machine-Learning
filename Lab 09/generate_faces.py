# generate_faces.py
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

z = torch.randn([1, G.z_dim]).to(device)
img = G(z, None, noise_mode='const')[0]
img = (img.clamp(-1, 1) + 1) * (255 / 2)
img = img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)

PIL.Image.fromarray(img).save("generated_outputs/realistic_face.png")
print("✅ Generated realistic face: 'generated_outputs/realistic_face.png'")
