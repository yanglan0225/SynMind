import os
import torch
import pickle
import numpy as np
from PIL import Image
from tqdm import tqdm
import shutil

# --- Configuration ---
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# --- Paths ---
# Directory where your saved .pkl files are
# Directory where the original images are storedrison
output_base_dir = '/home/yl/ssd_new/ymh/pami25/results/mix_v1/blurry'
# Path to the VAE model checkpoint
vae_ckpt_path = '/home/yl/ssd_new/ymh/pretrain_weights/mindeye2/sd_image_var_autoenc.pth'

# --- Parameters ---
# How many images per subject to check. Set to a high number (e.g., 99999) to check all.
num_images_to_check_per_subject = 1000
# The scaling factor used during encoding. We need to reverse this.
SCALING_FACTOR = 0.18215

# 1. Load the same VAE model used for encoding
print("Loading VAE model for decoding...")
from diffusers import AutoencoderKL
autoenc = AutoencoderKL(
    down_block_types=['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],
    up_block_types=['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'],
    block_out_channels=[128, 256, 512, 512],
    layers_per_block=2,
    sample_size=224,
)
try:
    ckpt = torch.load(vae_ckpt_path, map_location=device)
    autoenc.load_state_dict(ckpt)
    print("VAE model loaded successfully.")
except FileNotFoundError:
    print(f"ERROR: VAE checkpoint not found at {vae_ckpt_path}. Please check the path.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the VAE model: {e}")
    exit()

autoenc.eval()
autoenc.to(device)


def decode_and_save_images(subj):
    """
    Loads latents for a subject, decodes them back to images, and saves them.
    """
    print(f"\n--- Processing Subject {subj} ---")
    pred = torch.load(f"/home/yl/ssd_new/ymh/pami25/results/mix_v1/test_subj01.pth")
    prior_out = pred["vae_results"]
    labels = np.load("/home/ymh/adaptive_SemBrain/new_log/map/01_test_cocoid_map.npy")
    
    # --- Define paths for the current subject ---
    output_dir_subj = os.path.join(output_base_dir, f'subj{subj}')
    
    # Create the output directory
    os.makedirs(output_dir_subj, exist_ok=True)
    
    # Check if the latent file exists


    # --- Decode each latent and save the image ---
    count = 0
    for i in tqdm(range(1000)):
            
        # Reverse the scaling factor before decoding
        latents_tensor = prior_out[i] / SCALING_FACTOR
        
        # Decode the latent to get an image tensor
        with torch.no_grad():
            # The output is a sample from the distribution, in range [-1, 1]
            image_tensor = autoenc.decode(latents_tensor.unsqueeze(0)).sample[0] # Get the first image from the batch
        
        # Post-process the image tensor to be saveable
        # 1. Map from [-1, 1] to [0, 1]
        image = (image_tensor / 2 + 0.5).clamp(0, 1)
        # 2. Permute from (C, H, W) to (H, W, C)
        image = image.permute(1, 2, 0)
        # 3. Scale to [0, 255] and convert to uint8 numpy array
        image = (image * 255).round().to(torch.uint8).cpu().numpy()
        
        # Convert to PIL Image and save
        reconstructed_image = Image.fromarray(image)
        save_path = os.path.join("/home/yl/ssd_new/ymh/pami25/results/mix_v1/blurry/subj01", f"{labels[i]}.jpg")
        reconstructed_image.save(save_path)
    print(f"Finished processing for subject {subj}. Check results in: {output_dir_subj}")


# --- Main execution block ---
if __name__ == '__main__':
    subjects = ['01'] # The subjects you processed
    
    for subj_num in subjects:
        decode_and_save_images(subj_num)

    print(f"\nAll subjects processed. Verification images are saved in: {output_base_dir}")