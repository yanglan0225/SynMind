import os
import sys
import argparse
import numpy as np
from einops import rearrange
import time
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from dataloader_img import get_dataloader
# SDXL unCLIP requires code from https://github.com/Stability-AI/generative-models/tree/main
sys.path.append('generative_models/')
# tf32 data type is faster than standard float32
torch.backends.cuda.matmul.allow_tf32 = True
import gc
import utils
import psutil
utils.seed_everything(42)
print("PID of this process =",os.getpid())
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:",device)

DEFAULT_NSD_SUBJECTS = (1, 2, 5, 7)
NSD_SUBJECT_VOXELS = {
    1: 15724,
    2: 14278,
    5: 13039,
    7: 12682,
}


def parse_subjects(subjects):
    parsed_subjects = []
    for subject in subjects:
        for part in str(subject).split(","):
            subject_id = part.strip()
            if subject_id.lower().startswith("subj"):
                subject_id = subject_id[4:]
            if not subject_id.isdigit():
                raise ValueError(f"Invalid subject id: {subject}")
            subject_num = int(subject_id)
            if subject_num not in NSD_SUBJECT_VOXELS:
                raise ValueError(f"Unsupported NSD subject: {subject_num}")
            parsed_subjects.append(subject_num)
    return parsed_subjects


def print_cpu_memory_usage(tag=""):
    """Print the current process CPU memory usage."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    # rss is the resident set size in physical memory.
    rss_mb = mem_info.rss / (1024 * 1024) 
    print(f"[{tag}] Current CPU Memory Usage: {rss_mb:.2f} MB")


def parse_argument():
    parser = argparse.ArgumentParser(description="Model Training Configuration")
    parser.add_argument(
        "--model_name", type=str, default="v1",
        help="name of model, used for ckpt saving and wandb logging (if enabled)",
    )
    parser.add_argument(
        "--use_prior",action=argparse.BooleanOptionalAction,default=True,
        help="whether to train diffusion prior (True) or just rely on retrieval part of the pipeline (False)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=256,
        help="Batch size can be increased by 10x if only training retreival submodule and not diffusion prior",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=[str(subject) for subject in DEFAULT_NSD_SUBJECTS],
        help="NSD subjects to export. Defaults to the paper subjects: 1 2 5 7.",
    )
    parser.add_argument(
        "--mixup_pct",type=float,default=0.0,
        help="proportion of way through training when to switch from BiMixCo to SoftCLIP",
    )
    parser.add_argument(
        "--blurry_recon",action=argparse.BooleanOptionalAction,default=True,
        help="whether to output blurry reconstructions",
    )
    parser.add_argument(
        "--blur_scale",type=float,default=.5,
        help="multiply loss from blurry recons by this number",
    )
    parser.add_argument(
        "--clip_scale",type=float,default=1.,
        help="multiply contrastive loss by this number",
    )
    parser.add_argument(
        "--prior_scale",type=float,default=30,
        help="multiply diffusion prior loss by this",
    )
    parser.add_argument(
        "--epochs",type=int,default=80,
        help="number of epochs of training",
    )
    parser.add_argument(
        "--n_blocks",type=int,default=4,
    )
    parser.add_argument(
        "--hidden_dim",type=int,default=4096,
    )
    parser.add_argument(
        "--lr_scheduler_type",type=str,default='cycle',choices=['cycle','linear'],
    )
    parser.add_argument(
        "--ckpt_saving",action=argparse.BooleanOptionalAction,default=True,
    )
    parser.add_argument(
        "--ckpt_interval",type=int,default=10,
        help="save backup ckpt and reconstruct every x epochs",
    )
    parser.add_argument(
        "--max_lr",type=float,default=1e-4,
    )
    args = parser.parse_args()

    return args

    
# seed all random functions
class MindEyeModule(nn.Module):
    def __init__(self):
        super(MindEyeModule, self).__init__()
    def forward(self, x):
        return x
    
class SubjRidgeRegression(torch.nn.Module):
    def __init__(self, input_sizes, out_features): 
        super(SubjRidgeRegression, self).__init__()
        self.out_features = out_features
        self.linears = torch.nn.ModuleList([
                torch.nn.Linear(input_size, out_features) for input_size in input_sizes
            ])
    def forward(self, x, subj_idx):
        out = self.linears[subj_idx](x).unsqueeze(1)
        return out

SubjectWiseMapper = SubjRidgeRegression


def save_ckpt(tag,outdir,epoch, model, optimizer, lr_scheduler, losses, test_losses, lrs):
    ckpt_path = outdir+f'/{tag}.pth'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'train_losses': losses,
        'test_losses': test_losses,
        'lrs': lrs,
        }, ckpt_path)
    print(f"\n---saved {outdir}/{tag} ckpt!---\n")


def load_ckpt(tag, model, outdir, load_lr=False, load_optimizer=False, load_epoch=False, strict=False): 
    checkpoint_path = f"{outdir}/{tag}.pth"
    print(f"Loading from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    state_dict = checkpoint['model_state_dict']

    # Create a new state_dict excluding all keys associated with 'ridge'
    
    print("Loading model state dict...")
    # Using strict=False is important to ignore missing keys (like the ridge keys we just removed)
    model.load_state_dict(state_dict, strict=True) 
  
    if load_epoch:
        # It's safer to use .get() in case the key doesn't exist
        epoch = checkpoint.get('epoch')
        if epoch is not None:
            # You might need to make 'epoch' a global variable or handle it differently
            # depending on your script's structure.
            print("Epoch", epoch)

    if load_optimizer and 'optimizer_state_dict' in checkpoint:
        # Make sure optimizer is defined before calling this function
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if load_lr and 'lr_scheduler' in checkpoint:
        # Make sure lr_scheduler is defined before calling this function
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        
    del checkpoint, state_dict




def main(args):

    outdir = os.path.abspath(f'/root/autodl-tmp/train_logs/mix_mindeye_v1/{args.model_name}')
    if not os.path.exists(outdir) and args.ckpt_saving:
        os.makedirs(outdir,exist_ok=True)
    subjects = parse_subjects(args.subjects)
    subject_to_model_idx = {subject: idx for idx, subject in enumerate(subjects)}
    num_iterations_per_epoch = 108000 / args.batch_size # 108000 is the number of training samples in NSD
    
    model = MindEyeModule()

    # SWM: subject-wise mapper from subject-specific fMRI voxels to a shared latent space.
    model.ridge = SubjRidgeRegression([NSD_SUBJECT_VOXELS[subject] for subject in subjects], out_features=args.hidden_dim).to(args.device)


    #model.text_ridge = SubjRidgeRegression([15724,14278,13039,12682], out_features=args.hidden_dim).to(args.device)

    from models import BrainNetwork,sem_brainMLP
    # SSE + SSV: shared semantic and visual encoders over the SWM latent.
    model.backbone = BrainNetwork(h=args.hidden_dim, in_dim=args.hidden_dim, seq_len=1, n_blocks=args.n_blocks,
                            clip_size=1664, out_dim=1664*256, text_out_dim = 77*768,
                            blurry_recon=args.blurry_recon, clip_scale=args.clip_scale, use_text=True).to(args.device)
    
    print_cpu_memory_usage("After Backbone Init")
    
    if args.use_prior:
        from models import PriorNetwork, BrainDiffusionPrior,sem_PriorNetwork
        # setup diffusion prior network
        out_dim = 1664
        depth = 6
        dim_head = 52 #52
        heads = 32 #1664//52 # heads * dim_head = clip_emb_dim
        timesteps = 100

        prior_network = PriorNetwork(
                dim=out_dim,
                depth=depth,
                dim_head=dim_head,
                heads=heads,
                causal=False,
                num_tokens = 256,
                learned_query_mode="pos_emb"
            )
        print_cpu_memory_usage("After Prior Network Init")

        # SAR: visual diffusion prior for rendering image/unCLIP latents.
        model.diffusion_prior = BrainDiffusionPrior(
            net=prior_network,
            image_embed_dim=out_dim,
            condition_on_text_encodings=False,
            timesteps=timesteps,
            cond_drop_prob=0.2,
            image_embed_scale=None,
        ).to(args.device)
        print_cpu_memory_usage("After Prior Network Init")

        # setup diffusion prior network
        text_out_dim = 768
        depth = 4
        text_dim_head = 64
        text_heads = 8 # heads * dim_head = clip_emb_dim
        timesteps = 1000

        text_prior_network = sem_PriorNetwork(
                dim=text_out_dim,
                depth=depth,
                dim_head=text_dim_head,
                heads=text_heads,
                causal=False,
                num_tokens = 77
            )
        print_cpu_memory_usage("After Text Prior Network Init")

        # SAR: semantic diffusion prior for rendering text-token latents.
        model.text_diffusion_prior = BrainDiffusionPrior(
            net=text_prior_network,
            image_embed_dim=text_out_dim,
            condition_on_text_encodings=False,
            timesteps=timesteps,
            cond_drop_prob=0.2,
            image_embed_scale=None,
        ).to(args.device)
        load_ckpt("last",model,"/root/autodl-tmp/train_logs/mix_mindeye_v1/v2")
        model.to(args.device)
        

        print_cpu_memory_usage("After Text Prior Network Init")

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    opt_grouped_parameters = [
        {'params': [p for n, p in model.ridge.named_parameters()], 'weight_decay': 1e-2},

        {'params': [p for n, p in model.backbone.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
        {'params': [p for n, p in model.backbone.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        #{'params': [p for n, p in model.text_bacbone.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
        #{'params': [p for n, p in model.text_bacbone.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]
    if args.use_prior:
        opt_grouped_parameters.extend([
            {'params': [p for n, p in model.diffusion_prior.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
            {'params': [p for n, p in model.diffusion_prior.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            {'params': [p for n, p in model.text_diffusion_prior.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
            {'params': [p for n, p in model.text_diffusion_prior.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        ])

    optimizer = torch.optim.AdamW(opt_grouped_parameters, lr=args.max_lr)

    if args.lr_scheduler_type == 'linear':
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            total_iters=int(np.floor(args.epochs*num_iterations_per_epoch)),
            last_epoch=-1
        )
    elif args.lr_scheduler_type == 'cycle':
        total_steps=int(np.floor(args.epochs*num_iterations_per_epoch))
        print("total_steps", total_steps)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=args.max_lr,
            total_steps=total_steps,
            final_div_factor=1000,
            last_epoch=-1, pct_start=2/args.epochs
        )
            

    num_params = utils.count_params(model)
    gc.collect()

    print("-----------------------------------------------------")
    print("-----------------------------------------------------")
    print("-----------------------------------------------------")
    print_cpu_memory_usage("After Prior Network Init")


    logging.basicConfig(filename=f'/root/autodl-tmp/train_logs/mix_mindeye_v1/v1/train.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info(args)



    epoch = 0
    losses, test_losses, lrs = [], [], []
    best_test_loss = 1e9
    torch.cuda.empty_cache()
    train_loader = None
    progress_bar = tqdm(range(1), ncols=1200)
    test_image, test_voxel = None, None
    mse = nn.MSELoss()
    l1 = nn.L1Loss()
    soft_loss_temps = utils.cosine_anneal(0.004, 0.0075, args.epochs - int(args.mixup_pct * args.epochs))

    for epoch in progress_bar:
        model.eval()
        if (epoch+1) % 1 == 0:
            for export_subject in subjects:
                results = []
                vae_results = []
                text_results = []
                subject_model_idx = subject_to_model_idx[export_subject]
                with torch.no_grad(): 
                    train_loader = None
                    test_loader = get_dataloader(mode='test', batch_size=args.batch_size, subjects=[export_subject],clip_length=args.length)
                    for test_i, (voxel0, text_clip_data, subj_idx, image_enc, cnx_embeds, cnx_aug_embeds, clip_target) in enumerate(test_loader):  
                        # all test samples should be loaded per batch such that test_i should never exceed 0
                        text_clip_data = text_clip_data.to(args.device).float().view(text_clip_data.shape[0], 77, 768)
                        image_enc=image_enc.to(args.device).float()
                        cnx_embeds=cnx_embeds.to(args.device).float()
                        cnx_aug_embeds=cnx_aug_embeds.to(args.device).float()
                        clip_target = clip_target.to(args.device).float().view(clip_target.shape[0], 256, 1664)

                        voxel_ridge_list = []
                        for s_d in voxel0:
                            subj_shared = model.ridge(s_d.unsqueeze(0).to(args.device), subject_model_idx)
                            voxel_ridge_list.append(subj_shared)

                        voxel_ridge = torch.cat(voxel_ridge_list, dim=0)

                        backbone, clip_voxels, blurry_image_enc_, text_backbone = model.backbone(voxel_ridge)
                        text_backbone = text_backbone.view(text_backbone.shape[0], 77, 768)
                        b,_ = blurry_image_enc_
                        vae_results.append(b)

                        if args.use_prior:
                            text_prior_out = model.text_diffusion_prior.p_sample_loop(text_backbone.shape, 
                                            text_cond = dict(text_embed = text_backbone), 
                                            cond_scale = 1., timesteps = 50)                       
                            prior_out = model.diffusion_prior.p_sample_loop(backbone.shape, 
                                            text_cond = dict(text_embed = backbone), 
                                            cond_scale = 1., timesteps = 20)
                            results.append(prior_out)
                            text_results.append(text_prior_out)

                results = torch.cat(results, dim=0)
                vae_results = torch.cat(vae_results, dim=0)
                text_results = torch.cat(text_results, dim=0)
                torch.save({
                    'results': results,
                    'vae_results': vae_results,
                    'text_results': text_results,
                }, f'{outdir}/test_subj{export_subject:02d}_10e.pth')

if __name__=='__main__':
    args = parse_argument()
    args.length = 75
    args.data_type = torch.float16
    args.device = device
    main(args)
        
