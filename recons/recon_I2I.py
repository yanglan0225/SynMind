import sys
import numpy as np
import torch
from torchvision import transforms
# SDXL unCLIP requires code from https://github.com/Stability-AI/generative-models/tree/main
sys.path.append('generative_models/')
from generative_models.sgm.models.diffusion import DiffusionEngine
from omegaconf import OmegaConf
from tqdm import tqdm
# tf32 data type is faster than standard float32
torch.backends.cuda.matmul.allow_tf32 = True
# custom functions #
import utils_recon
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("device:",device)
# seed all random functions
utils_recon.seed_everything(42)
    # prep unCLIP
print("Loading unCLIP model...")
config = OmegaConf.load("generative_models/configs/unclip6.yaml")
config = OmegaConf.to_container(config, resolve=True)
unclip_params = config["model"]["params"]
network_config = unclip_params["network_config"]
denoiser_config = unclip_params["denoiser_config"]
first_stage_config = unclip_params["first_stage_config"]
conditioner_config = unclip_params["conditioner_config"]
sampler_config = unclip_params["sampler_config"]
scale_factor = unclip_params["scale_factor"]
disable_first_stage_autocast = unclip_params["disable_first_stage_autocast"]
offset_noise_level = unclip_params["loss_fn_config"]["params"]["offset_noise_level"]

first_stage_config['target'] = 'sgm.models.autoencoder.AutoencoderKL'
sampler_config['params']['num_steps'] = 38

diffusion_engine = DiffusionEngine(network_config=network_config,
                    denoiser_config=denoiser_config,
                    first_stage_config=first_stage_config,
                    conditioner_config=conditioner_config,
                    sampler_config=sampler_config,
                    scale_factor=scale_factor,
                    disable_first_stage_autocast=disable_first_stage_autocast)
# set to inference
diffusion_engine.eval().requires_grad_(False)
diffusion_engine.to(device)
ckpt_path = f'/home/yl/ssd_new/ymh/pretrain_weights/mindeye2/unclip6_epoch0_step110000.ckpt'
ckpt = torch.load(ckpt_path, map_location='cpu')
diffusion_engine.load_state_dict(ckpt['state_dict'])

batch={"jpg": torch.randn(1,3,1,1).to(device), # jpg doesnt get used, it's just a placeholder
    "original_size_as_tuple": torch.ones(1, 2).to(device) * 768,
    "crop_coords_top_left": torch.zeros(1, 2).to(device)}
out = diffusion_engine.conditioner(batch)
vector_suffix = out["vector"].to(device)
print("vector_suffix", vector_suffix.shape)
# Load hdf5 data for betas
for subj in [1,2,5,7]:
    pred = torch.load(f"/home/yl/ssd_new/ymh/pami25/weights/vi/test_subj0{subj}_10e.pth")
    pred = pred["results"]
    print(pred.shape)
    prior_out = pred
    print(prior_out.shape)

    # labels = np.load("/home/ymh/adaptive_SynMind/new_log/map/01_test_cocoid_map.npy")
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
        for i in tqdm(range(0,982)):
            samples = utils_recon.unclip_recon(prior_out[i].unsqueeze(0),
                            diffusion_engine,
                            vector_suffix,
                            num_samples=1)
            img =   transforms.ToPILImage()(samples[0])
            img.save(f"/home/yl/ssd_new/ymh/pami25/results/subj0{subj}/I2I_vi/{i}.jpg")
