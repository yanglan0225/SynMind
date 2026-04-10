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
        "--model_name", type=str, default="subj01",
        help="name of model, used for ckpt saving and wandb logging (if enabled)",
    )
    parser.add_argument(
        "--use_prior",action=argparse.BooleanOptionalAction,default=True,
        help="whether to train diffusion prior (True) or just rely on retrieval part of the pipeline (False)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=48,
        help="Batch size can be increased by 10x if only training retreival submodule and not diffusion prior",
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
        "--epochs",type=int,default=60,
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
        "--ckpt_interval",type=int,default=5,
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

import torch
'''
def load_ckpt(tag, model,optimizer, outdir, epoch,lr_scheduler,load_lr=True, load_optimizer=True, load_epoch=True, strict=False): 
    print(f"\n---loading {outdir}/{tag}.pth ckpt---\n")
    # Note: the original code loaded 'last.pth', but the function receives a 'tag' argument.
    # Using the 'tag' argument makes the function more flexible.
    checkpoint_path = f"/root/autodl-tmp/train_logs/mix_mindeye_v1/v2/last.pth"
    print(f"Loading from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    state_dict = checkpoint['model_state_dict']
    
    print("Loading model state dict...")
    # Using strict=False is important to ignore missing keys (like the ridge keys we just removed)
    model.load_state_dict(state_dict, strict=True) 
    print("Model loaded successfully, ignoring 'ridge' weights.")

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
        print(f"Type of checkpoint['lr_scheduler']: {type(checkpoint['lr_scheduler'])}")
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        
    del checkpoint, state_dict
    return epoch, lr_scheduler
'''

def load_ckpt(tag, model, outdir, 
            load_lr=False, load_optimizer=False, load_epoch=False, strict=False,
            load_ridge_custom=True,
            subj01_ckpt_path="/root/autodl-tmp/pretrained_weights/mindeyev2/train_logs/final_subj01_pretrained_40sess_24bs/last.pth"
           ): 
    """Load a checkpoint and optionally load custom ridge weights."""
    
    main_ckpt_path = f"{outdir}/{tag}.pth"
    print(f"\n--- Loading main checkpoint: {main_ckpt_path} ---")
    
    try:
        checkpoint = torch.load(main_ckpt_path, map_location='cpu')
        state_dict = checkpoint['model_state_dict']
    except Exception as e:
        print(f"Error: failed to load main checkpoint: {e}")
        return

    print("Loading non-'ridge' weights...")
    filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('ridge.')}
    model.load_state_dict(state_dict, strict=True) 
    if load_epoch:
        epoch = checkpoint.get('epoch')
        if epoch is not None:
            print("\nEpoch", epoch)

    if load_optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if load_lr and 'lr_scheduler' in checkpoint:
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        
    del checkpoint, state_dict, filtered_state_dict

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

'''
def load_ckpt(tag, model, outdir, load_lr=False, load_optimizer=False, load_epoch=False, strict=False): 
    print(f"\n---loading {outdir}/v1_30_last.pth ckpt---\n")
    # Note: the original code loaded 'last.pth', but the function receives a 'tag' argument.
    # Using the 'tag' argument makes the function more flexible.
    checkpoint_path = f"{outdir}/{tag}.pth"
    print(f"Loading from: {checkpoint_path}")
    checkpoint = torch.load("/root/autodl-tmp/train_logs/mix_mindeye_v1/v2/last.pth", map_location='cpu')
    
    state_dict = checkpoint['model_state_dict']

    # Create a new state_dict excluding all keys associated with 'ridge'
    
    print("Loading model state dict...")
    # Using strict=False is important to ignore missing keys (like the ridge keys we just removed)
    model.load_state_dict(state_dict, strict=False) 
  
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
'''
def load_sem_ckpt( model): 
    checkpoint = torch.load('/root/autodl-tmp/train_logs/iccv_Wblurry_75_v1/model_checkpoint_epoch20.pth', map_location='cpu')
    print(checkpoint.keys())
    #adjusted_state_dict = {key.replace("ridge.", ""): value for key, value in checkpoint["net_state_dict"].items()}
    model.text_bacbone.load_state_dict(checkpoint["net_state_dict"], strict=False)
    #mem_model.load_state_dict(checkpoint["mem_state_dict"])
    #adjusted_state_dict = {key.replace("ridge.", ""): value for key, value in checkpoint["model_state_dict"].items()}
    #model.text_ridge.load_state_dict(checkpoint["model_state_dict"])
    model.text_diffusion_prior.load_state_dict(checkpoint["prior_state_dict"])
        
    del checkpoint
    print("Model loaded successfully, ignoring 'ridge' weights.")


def main(args):

    outdir = os.path.abspath(f'/root/autodl-tmp/train_logs/mix_mindeye_v1/{args.model_name}')
    if not os.path.exists(outdir) and args.ckpt_saving:
        os.makedirs(outdir,exist_ok=True)
    num_iterations_per_epoch = 54000 / args.batch_size # 108000 is the number of training samples in NSD
    
    model = MindEyeModule()
    print_cpu_memory_usage("Init")

    model.ridge = SubjRidgeRegression([15724,14278,13039,12682], out_features=args.hidden_dim).to(args.device)
    #model.text_ridge = SubjRidgeRegression([15724,14278,13039,12682], out_features=args.hidden_dim).to(args.device)

    from models import BrainNetwork,sem_brainMLP
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

        model.diffusion_prior = BrainDiffusionPrior(
            net=prior_network,
            image_embed_dim=out_dim,
            condition_on_text_encodings=False,
            timesteps=timesteps,
            cond_drop_prob=0.2,
            image_embed_scale=None,
        ).to(args.device)
        print_cpu_memory_usage("After Prior Network Init")
        print_cpu_memory_usage("After Prior Network Init")

        #load_ckpt("last",model,"/root/autodl-tmp/pretrained_weights/mindeyev2/train_logs/final_subj01_pretrained_40sess_24bs")
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

        model.text_diffusion_prior = BrainDiffusionPrior(
            net=text_prior_network,
            image_embed_dim=text_out_dim,
            condition_on_text_encodings=False,
            timesteps=timesteps,
            cond_drop_prob=0.2,
            image_embed_scale=None,
        ).to(args.device)
        #model.text_bacbone = sem_brainMLP(hidden_dim=args.hidden_dim)
        model.to(args.device)

        load_ckpt("last",model,"/root/autodl-tmp/train_logs/mix_mindeye_v1/v2")
        

        print_cpu_memory_usage("After Text Prior Network Init")

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    opt_grouped_parameters = [
        {'params': [p for n, p in model.ridge.named_parameters()], 'weight_decay': 1e-2},
        #{'params': [p for n, p in model.text_ridge.named_parameters()], 'weight_decay': 1e-2},
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
        print("Using OneCycleLR with max_lr:", args.max_lr)
            

    num_params = utils.count_params(model)
    gc.collect()

    print("-----------------------------------------------------")
    print("-----------------------------------------------------")
    print("-----------------------------------------------------")
    print_cpu_memory_usage("After Prior Network Init")


    logging.basicConfig(filename=f'/root/autodl-tmp/train_logs/mix_mindeye_v1/{args.model_name}/train.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info(args)



    epoch = 0
    losses, test_losses, lrs = [], [], []
    best_test_loss = 1e9
    torch.cuda.empty_cache()
    train_loader = None
    test_image, test_voxel = None, None
    mse = nn.MSELoss()
    l1 = nn.L1Loss()
    soft_loss_temps = utils.cosine_anneal(0.004, 0.0075, args.epochs - int(args.mixup_pct * args.epochs))
    print(lr_scheduler)
    print(f"Type of checkpoint['lr_scheduler']: {type(lr_scheduler)}")
    print(epoch)
    epoch = epoch
    progress_bar = tqdm(range(epoch,args.epochs), ncols=1200)

    
    for epoch in progress_bar:
       
        if epoch % 2 == 0:
            current_part = 'first'
        else:
            current_part = 'second'
        
        print(f"\n{'='*20} Epoch {epoch} {'='*20}")
        print(f"--- Creating new DataLoader for the '{current_part}' half of the data ---")
        if train_loader is not None:
            del train_loader
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        train_loader = get_dataloader(
            mode='train', 
            batch_size=args.batch_size, 
            subjects=[1,2,5,7],
            clip_length=args.length,
            data_part = current_part,
        )       


        model.train()

        fwd_percent_correct = 0.
        bwd_percent_correct = 0.
        test_fwd_percent_correct = 0.
        test_bwd_percent_correct = 0.
        
        recon_cossim = 0.
        test_recon_cossim = 0.
        recon_mse = 0.
        test_recon_mse = 0.

        loss_clip_total = 0.
        text_loss_clip_total = 0.
        loss_blurry_total = 0.
        loss_blurry_cont_total = 0.
        test_loss_clip_total = 0.
        test_text_loss_clip_total = 0.
        
        loss_prior_total = 0.
        text_loss_prior_total = 0.
        test_loss_prior_total = 0.
        test_text_loss_prior_total = 0.

        blurry_pixcorr = 0.
        test_blurry_pixcorr = 0. # needs >.456 to beat low-level subj01 results in mindeye v1

                

        for  train_i,data in enumerate(train_loader):

            if True:
                optimizer.zero_grad()
                loss=0.
                voxel0, text_clip_data, subj_idx,class_labels, image_enc, cnx_embeds, cnx_aug_embeds, clip_target, clip_emb= data
                
                text_clip_data = text_clip_data.to(args.device).float().view(text_clip_data.shape[0], 77, 768)
                image_enc=image_enc.to(args.device).float().view(-1,4,28,28)
                cnx_embeds=cnx_embeds.to(args.device).float()
                cnx_aug_embeds=cnx_aug_embeds.to(args.device).float()
                clip_target = clip_target.to(args.device).float().view(clip_target.shape[0], 256, 1664)
                #clip_emb = clip_emb.to(args.device).float().view(clip_emb.shape[0], 768)

                assert not torch.any(torch.isnan(clip_target))

                if epoch < int(args.mixup_pct * args.epochs):
                    voxel0, perm, betas, select = utils.mixco(voxel0)
                    perm_list = [perm_iters[f"subj0{s}_iter{train_i}"].detach().to(device) for s in subj_list]
                    perm = torch.cat(perm_list, dim=0)
                    betas_list = [betas_iters[f"subj0{s}_iter{train_i}"].detach().to(device) for s in subj_list]
                    betas = torch.cat(betas_list, dim=0)
                    select_list = [select_iters[f"subj0{s}_iter{train_i}"].detach().to(device) for s in subj_list]
                    select = torch.cat(select_list, dim=0)

                voxel_ridge_list = []
                text_voxel_ridge_list = []
                for s_d,subj in zip(voxel0,subj_idx):
                    subj_shared = model.ridge(s_d.unsqueeze(0).to(args.device), subj)
                    #subj_shared_text = model.text_ridge(s_d.unsqueeze(0).to(args.device), subj)
                    voxel_ridge_list.append(subj_shared)
                    #text_voxel_ridge_list.append(subj_shared_text)

                voxel_ridge = torch.cat(voxel_ridge_list, dim=0)
                #text_voxel_ridge = torch.cat(text_voxel_ridge_list, dim=0)

                backbone, clip_voxels, blurry_image_enc_, text_backbone= model.backbone(voxel_ridge)
                #text_backbone = model.text_bacbone(voxel_ridge)

                if args.clip_scale>0:
                    clip_voxels_norm = nn.functional.normalize(clip_voxels.flatten(1), dim=-1)
                    clip_target_norm = nn.functional.normalize(clip_target.flatten(1), dim=-1)
                if args.clip_scale>0:
                    text_backbone_norm = nn.functional.normalize(text_backbone.flatten(1), dim=-1)
                    text_clip_data_norm = nn.functional.normalize(text_clip_data.flatten(1), dim=-1)
                    #text_emb_norm = nn.functional.normalize(text_emb.flatten(1), dim=-1)
                    #clip_emb_norm = nn.functional.normalize(clip_emb.flatten(1), dim=-1)

                if args.use_prior:
                    loss_prior, prior_out = model.diffusion_prior(text_embed=backbone, image_embed=clip_target)
                    text_loss_prior, prior_text_out = model.text_diffusion_prior(text_embed=text_backbone.view(-1,77,768), image_embed=text_clip_data)
                    loss_prior_total += loss_prior.item()
                    text_loss_prior_total += text_loss_prior.item()
                    loss_prior *= args.prior_scale
                    text_loss_prior *= args.prior_scale
                    loss += loss_prior
                    loss += text_loss_prior

                    recon_cossim += nn.functional.cosine_similarity(prior_out, clip_target).mean().item()
                    recon_mse += mse(prior_out, clip_target).item()


                if args.clip_scale>0:
                    if epoch < int(args.mixup_pct * args.epochs):                
                        loss_clip = utils.mixco_nce(
                            clip_voxels_norm,
                            clip_target_norm,
                            temp=.006,
                            perm=perm, betas=betas, select=select)
                    else:
                        epoch_temp = soft_loss_temps[epoch-int(args.mixup_pct*args.epochs)]
                        loss_clip = utils.soft_clip_loss(
                            clip_voxels_norm,
                            clip_target_norm,
                            temp=epoch_temp)
                        text_loss_clip = utils.soft_clip_loss(
                            text_backbone_norm,
                            text_clip_data_norm,
                            temp=epoch_temp,bi=False)
                        #text_emb_loss_clip = utils.soft_clip_loss(
                        #    text_emb_norm,
                        #    clip_emb_norm,
                        #    temp=epoch_temp,bi=False)

                    loss_clip_total += loss_clip.item()
                    text_loss_clip_total += text_loss_clip.item()
                    loss_clip *= args.clip_scale
                    text_loss_clip *= args.clip_scale
                    #text_emb_loss_clip *= args.clip_scale
                    loss += loss_clip
                    loss += text_loss_clip
                    #loss += text_emb_loss_clip

                if args.blurry_recon:     
                    image_enc_pred, transformer_feats = blurry_image_enc_
                    loss_blurry = l1(image_enc_pred, image_enc)
                    loss_blurry_total += loss_blurry.item()

                    if epoch < int(args.mixup_pct * args.epochs):
                        image_enc_shuf = image_enc[perm]
                        betas_shape = [-1] + [1]*(len(image_enc.shape)-1)
                        image_enc[select] = image_enc[select] * betas[select].reshape(*betas_shape) + \
                            image_enc_shuf[select] * (1 - betas[select]).reshape(*betas_shape)


                    cont_loss = utils.soft_cont_loss(
                        nn.functional.normalize(transformer_feats.reshape(-1, transformer_feats.shape[-1]), dim=-1),
                        nn.functional.normalize(cnx_embeds.reshape(-1, transformer_feats.shape[-1]), dim=-1),
                        nn.functional.normalize(cnx_aug_embeds.reshape(-1, transformer_feats.shape[-1]), dim=-1),
                        temp=0.2)
                    loss_blurry_cont_total += cont_loss.item()

                    loss += (loss_blurry + 0.1*cont_loss) * args.blur_scale #/.18215

                if args.clip_scale>0:
                    # forward and backward top 1 accuracy        
                    labels = torch.arange(len(clip_voxels_norm)).to(clip_voxels_norm.device) 
                    fwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_voxels_norm, clip_target_norm), labels, k=1).item()
                    bwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_target_norm, clip_voxels_norm), labels, k=1).item()


                utils.check_loss(loss)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # gradient clipping
                
                optimizer.step()

                losses.append(loss.item())
                lrs.append(optimizer.param_groups[0]['lr'])

                if args.lr_scheduler_type is not None:
                    lr_scheduler.step()
                if (train_i+1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{args.epochs}, Iteration {train_i+1}/{len(train_loader)}, Loss: {loss.item():.4f}, loss_prior: {loss_prior.item():.4f}, text_loss_prior: {text_loss_prior.item():.4f}, loss_clip: {loss_clip.item():.4f}, text_loss_clip: {text_loss_clip.item():.4f}, loss_blurry: {loss_blurry.item():.4f}, loss_blurry_cont: {0.1*cont_loss.item():.4f}, recon_cossim: {recon_cossim/(train_i+1):.4f}, recon_mse: {recon_mse/(train_i+1):.4f}, lr: {optimizer.param_groups[0]['lr']:.6f}")
                    logging.info(f"Epoch {epoch+1}/{args.epochs}, Iteration {train_i+1}/{len(train_loader)}, Loss: {loss.item():.4f}, loss_prior: {loss_prior.item():.4f}, text_loss_prior: {text_loss_prior.item():.4f}, loss_clip: {loss_clip.item():.4f}, text_loss_clip: {text_loss_clip.item():.4f}, loss_blurry: {loss_blurry.item():.4f}, loss_blurry_cont: {0.1*cont_loss.item():.4f}, recon_cossim: {recon_cossim/(train_i+1):.4f}, recon_mse: {recon_mse/(train_i+1):.4f}, lr: {optimizer.param_groups[0]['lr']:.6f}")

        model.eval()
        if (epoch+1) % 1 == 0:
            loss = 0.
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=args.data_type): 
                train_loader = None
                test_loader = get_dataloader(mode='test', batch_size=args.batch_size, subjects=[1],clip_length=args.length)
                for test_i, (voxel0, text_clip_data, subj_idx,class_labels, image_enc, cnx_embeds, cnx_aug_embeds, clip_target,clip_emb) in enumerate(test_loader):  
                    # all test samples should be loaded per batch such that test_i should never exceed 0
                    text_clip_data = text_clip_data.to(args.device).float().view(text_clip_data.shape[0], 77, 768)
                    image_enc=image_enc.to(args.device).float()
                    cnx_embeds=cnx_embeds.to(args.device).float()
                    cnx_aug_embeds=cnx_aug_embeds.to(args.device).float()
                    clip_target = clip_target.to(args.device).float().view(clip_target.shape[0], 256, 1664)
                    #clip_emb = clip_emb.to(args.device).float().view(clip_emb.shape[0], 1280)

                    voxel_ridge_list = []
                    text_voxel_ridge_list = []
                    for s_d,subj in zip(voxel0,subj_idx):
                        subj_shared = model.ridge(s_d.unsqueeze(0).to(args.device), subj)
                        #subj_shared_text = model.text_ridge(s_d.unsqueeze(0).to(args.device), subj)
                        voxel_ridge_list.append(subj_shared)
                        #text_voxel_ridge_list.append(subj_shared_text)

                    voxel_ridge = torch.cat(voxel_ridge_list, dim=0)
                    #text_voxel_ridge = torch.cat(text_voxel_ridge_list, dim=0)

                    backbone, clip_voxels, blurry_image_enc_, text_backbone= model.backbone(voxel_ridge)
                    #text_backbone = model.text_bacbone(voxel_ridge)

                    if args.clip_scale>0:
                        clip_voxels_norm = nn.functional.normalize(clip_voxels.flatten(1), dim=-1)
                        clip_target_norm = nn.functional.normalize(clip_target.flatten(1), dim=-1)
                        text_backbone_norm = nn.functional.normalize(text_backbone.flatten(1), dim=-1)
                        text_clip_data_norm = nn.functional.normalize(text_clip_data.flatten(1), dim=-1)

                    
                    # for some evals, only doing a subset of the samples per batch because of computational cost
                    random_samps = np.random.choice(np.arange(len(voxel0)), size=len(voxel0)//5, replace=False)
                    
                    if args.use_prior:
                        loss_prior, contaminated_prior_out = model.diffusion_prior(text_embed=backbone[random_samps], image_embed=clip_target[random_samps])
                        text_loss_prior, prior_text_out = model.text_diffusion_prior(text_embed=text_backbone[random_samps].view(-1,77,768), image_embed=text_clip_data[random_samps])
                        test_loss_prior_total += loss_prior.item()
                        test_text_loss_prior_total += text_loss_prior.item()
                        loss_prior *= args.prior_scale
                        text_loss_prior *= args.prior_scale
                        loss += loss_prior
                        loss += text_loss_prior
                        
                    if args.clip_scale>0:
                        loss_clip = utils.soft_clip_loss(
                            clip_voxels_norm,
                            clip_target_norm,
                            temp=.006)
                        text_loss_clip = utils.soft_clip_loss(
                            text_backbone_norm,
                            text_clip_data_norm,
                            temp=.006,
                            bi=False)
                        test_loss_clip_total += loss_clip.item()
                        test_loss_clip_total += loss_clip.item()
                        loss_clip = loss_clip * args.clip_scale
                        text_loss_clip = text_loss_clip * args.clip_scale
                        loss += loss_clip
                        loss += text_loss_clip

                    if args.clip_scale>0:
                        # forward and backward top 1 accuracy        
                        labels = torch.arange(len(clip_voxels_norm)).to(clip_voxels_norm.device) 
                        test_fwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_voxels_norm, clip_target_norm), labels, k=1).item()
                        test_bwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_target_norm, clip_voxels_norm), labels, k=1).item()
                    
                    utils.check_loss(loss)                
                    test_losses.append(loss.item())

                logs = {"train/loss": np.mean(losses[-(train_i+1):]),
                    "test/loss": np.mean(test_losses[-(test_i+1):]),
                    "train/lr": lrs[-1],
                    "train/num_steps": len(losses),
                    "test/num_steps": len(test_losses),
                    "train/fwd_pct_correct": fwd_percent_correct / (train_i + 1),
                    "train/bwd_pct_correct": bwd_percent_correct / (train_i + 1),
                    "test/test_fwd_pct_correct": test_fwd_percent_correct / (test_i + 1),
                    "test/test_bwd_pct_correct": test_bwd_percent_correct / (test_i + 1),
                    "train/loss_clip_total": loss_clip_total / (train_i + 1),
                    "train/text_loss_clip_total": text_loss_clip_total / (train_i + 1),
                    "train/loss_blurry_total": loss_blurry_total / (train_i + 1),
                    "train/loss_blurry_cont_total": loss_blurry_cont_total / (train_i + 1),
                    "test/loss_clip_total": test_loss_clip_total / (test_i + 1),
                    "test/text_loss_clip_total": test_text_loss_clip_total / (test_i + 1),
                    "train/blurry_pixcorr": blurry_pixcorr / (train_i + 1),
                    "test/blurry_pixcorr": test_blurry_pixcorr / (test_i + 1),
                    "train/recon_cossim": recon_cossim / (train_i + 1),
                    "test/recon_cossim": test_recon_cossim / (test_i + 1),
                    "train/recon_mse": recon_mse / (train_i + 1),
                    "test/recon_mse": test_recon_mse / (test_i + 1),
                    "train/loss_prior": loss_prior_total / (train_i + 1),
                    "train/text_loss_prior": text_loss_prior_total / (train_i + 1),
                    "test/loss_prior": test_loss_prior_total / (test_i + 1),
                    "test/text_loss_prior": test_text_loss_prior_total / (test_i + 1),
                    }
                print(logs)
                logging.info(logs)
            del test_loader
                
        # Save model checkpoint and reconstruct
        if (args.ckpt_saving) and ((epoch+1) % args.ckpt_interval == 0):
            save_ckpt(f"last",outdir,epoch, model, optimizer, lr_scheduler, losses, test_losses, lrs)

        torch.cuda.empty_cache()

    print("\n===Finished!===\n")
    if args.ckpt_saving:
        save_ckpt("last",outdir,epoch, model, optimizer, lr_scheduler, losses, test_losses, lrs)

if __name__=='__main__':
    args = parse_argument()
    args.length = 75
    args.data_type = torch.float16
    args.device = device
    main(args)
        
