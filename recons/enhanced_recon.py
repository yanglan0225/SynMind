import os
import sys
import json
import argparse
import numpy as np
import math
from einops import rearrange
import time
import random
import string
import h5py
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
from accelerate import Accelerator, DeepSpeedPlugin

# SDXL unCLIP requires code from https://github.com/Stability-AI/generative-models/tree/main
sys.path.append('generative_models/')
import sgm
from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder, FrozenCLIPEmbedder, FrozenOpenCLIPEmbedder2
from generative_models.sgm.models.diffusion import DiffusionEngine
from generative_models.sgm.util import append_dims
from omegaconf import OmegaConf

# tf32 data type is faster than standard float32
torch.backends.cuda.matmul.allow_tf32 = True

# custom functions #

accelerator = Accelerator(split_batches=False, mixed_precision="fp16")
device = "cuda"
print("device:",device)



def load_images_from_folder(folder, labels):

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ToTensor(),       
    ])
    
    images = []
    for label in labels:

        img_path = os.path.join(folder, f"{label}.jpg")
        
        if os.path.exists(img_path):
            img = Image.open(img_path).convert('RGB')  
            img = preprocess(img)  
            images.append(img)    
        else:
            print(f"Warning: Image file not found for label {label} at {img_path}")
    
    return torch.stack(images)  

labels = np.load("/home/ymh/adaptive_SynMind/new_log/map/01_test_cocoid_map.npy")

config = OmegaConf.load("generative_models/configs/unclip6.yaml")
config = OmegaConf.to_container(config, resolve=True)
unclip_params = config["model"]["params"]
sampler_config = unclip_params["sampler_config"]
sampler_config['params']['num_steps'] = 38
config = OmegaConf.load("generative_models/configs/inference/sd_xl_base.yaml")
config = OmegaConf.to_container(config, resolve=True)
refiner_params = config["model"]["params"]

network_config = refiner_params["network_config"]
denoiser_config = refiner_params["denoiser_config"]
first_stage_config = refiner_params["first_stage_config"]
conditioner_config = refiner_params["conditioner_config"]
scale_factor = refiner_params["scale_factor"]
disable_first_stage_autocast = refiner_params["disable_first_stage_autocast"]
# base_ckpt_path = '/weka/robin/projects/stable-research/checkpoints/sd_xl_base_1.0.safetensors'
base_ckpt_path = '/home/yl/ssd_new/ymh/pretrain_weights/mindeye2/zavychromaxl_v30.safetensors'
base_engine = DiffusionEngine(network_config=network_config,
                       denoiser_config=denoiser_config,
                       first_stage_config=first_stage_config,
                       conditioner_config=conditioner_config,
                       sampler_config=sampler_config, # using the one defined by the unclip
                       scale_factor=scale_factor,
                       disable_first_stage_autocast=disable_first_stage_autocast,
                       ckpt_path=base_ckpt_path)
base_engine.eval().requires_grad_(False)
base_engine.to(device)
base_text_embedder1 = FrozenCLIPEmbedder(
    layer=conditioner_config['params']['emb_models'][0]['params']['layer'],
    layer_idx=conditioner_config['params']['emb_models'][0]['params']['layer_idx'],
)
base_text_embedder1.to(device)

base_text_embedder2 = FrozenOpenCLIPEmbedder2(
    arch=conditioner_config['params']['emb_models'][1]['params']['arch'],
    version=conditioner_config['params']['emb_models'][1]['params']['version'],
    freeze=conditioner_config['params']['emb_models'][1]['params']['freeze'],
    layer=conditioner_config['params']['emb_models'][1]['params']['layer'],
    always_return_pooled=conditioner_config['params']['emb_models'][1]['params']['always_return_pooled'],
    legacy=conditioner_config['params']['emb_models'][1]['params']['legacy'],
)
base_text_embedder2.to(device)
batch={"txt": "",
      "original_size_as_tuple": torch.ones(1, 2).to(device) * 768,
      "crop_coords_top_left": torch.zeros(1, 2).to(device),
      "target_size_as_tuple": torch.ones(1, 2).to(device) * 1024}

out = base_engine.conditioner(batch)
crossattn = out["crossattn"].to(device)
vector_suffix = out["vector"][:,-1536:].to(device)
print("crossattn", crossattn.shape)
print("vector_suffix", vector_suffix.shape)
print("---")

batch_uc={"txt": "painting, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, deformed, ugly, blurry, bad anatomy, bad proportions, extra limbs, cloned face, skinny, glitchy, double torso, extra arms, extra hands, mangled fingers, missing lips, ugly face, distorted face, extra legs, anime",
      "original_size_as_tuple": torch.ones(1, 2).to(device) * 768,
      "crop_coords_top_left": torch.zeros(1, 2).to(device),
      "target_size_as_tuple": torch.ones(1, 2).to(device) * 1024}
out = base_engine.conditioner(batch_uc)
crossattn_uc = out["crossattn"].to(device)
vector_uc = out["vector"].to(device)
print("crossattn_uc", crossattn_uc.shape)
print("vector_uc", vector_uc.shape)
num_samples = 1 # PS: I tried increasing this to 16 and picking highest cosine similarity like we did in MindEye1, it didnt seem to increase eval performance!
img2img_timepoint = 13 # 9 # higher number means more reliance on prompt, less reliance on matching the conditioning image
base_engine.sampler.guider.scale = 5 # 5 # cfg
def denoiser(x, sigma, c): return base_engine.denoiser(base_engine.model, x, sigma, c)


all_enhancedrecons = None
for subj in [1,2,5,7]:
    all_recons = load_images_from_folder(f"/home/ymh/LG_feature_aligin/codes/train/fuse_v1/images", labels)

    all_predcaptions = torch.load(f"/home/yl/ssd_new/ymh/pami25/results/subj0{subj}_pred/all_predcaptions.pt")
    all_recons = transforms.Resize((768,768))(all_recons).float()
    os.makedirs(f"/home/yl/ssd_new/ymh/pami25/results/subj0{subj}_pred/I2I_E", exist_ok=True)
    output_dir = f"/home/yl/ssd_new/ymh/image-aligned-experiment-data/images/mindAliner/subj0{subj}_1E"
    os.makedirs(output_dir, exist_ok=True)
    for img_idx in tqdm(range(len(all_recons))):
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16), base_engine.ema_scope():
            base_engine.sampler.num_steps = 25
            
            image = all_recons[[img_idx]]
            
            image = image.to(device)
            prompt = all_predcaptions[[img_idx]][0]
            # prompt = ""

            # z = torch.randn(num_samples,4,96,96).to(device)
            assert image.shape[-1]==768
            z = base_engine.encode_first_stage(image*2-1).repeat(num_samples,1,1,1)

            openai_clip_text = base_text_embedder1(prompt)
            clip_text_tokenized, clip_text_emb  = base_text_embedder2(prompt)
            clip_text_emb = torch.hstack((clip_text_emb, vector_suffix))
            clip_text_tokenized = torch.cat((openai_clip_text, clip_text_tokenized),dim=-1)
            c = {"crossattn": clip_text_tokenized.repeat(num_samples,1,1), "vector": clip_text_emb.repeat(num_samples,1)}
            uc = {"crossattn": crossattn_uc.repeat(num_samples,1,1), "vector": vector_uc.repeat(num_samples,1)}

            noise = torch.randn_like(z)
            sigmas = base_engine.sampler.discretization(base_engine.sampler.num_steps).to(device)
            init_z = (z + noise * append_dims(sigmas[-img2img_timepoint], z.ndim)) / torch.sqrt(1.0 + sigmas[0] ** 2.0)
            sigmas = sigmas[-img2img_timepoint:].repeat(num_samples,1)

            base_engine.sampler.num_steps = sigmas.shape[-1] - 1
            noised_z, _, _, _, c, uc = base_engine.sampler.prepare_sampling_loop(init_z, cond=c, uc=uc, 
                                                                num_steps=base_engine.sampler.num_steps)
            for timestep in range(base_engine.sampler.num_steps):
                noised_z = base_engine.sampler.sampler_step(sigmas[:,timestep],
                                                            sigmas[:,timestep+1],
                                                            denoiser, noised_z, cond=c, uc=uc, gamma=0)
            samples_z_base = noised_z
            samples_x = base_engine.decode_first_stage(samples_z_base)
            samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)

            # find best sample
        
            samples = samples
            img = transforms.Resize((224,224))(samples).float()
            img =   transforms.ToPILImage()(samples[0])
            img.save(f"{output_dir}/{labels[img_idx]}.jpg")
