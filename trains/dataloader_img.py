import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class NSDDataset(Dataset):
    def __init__(self, data_root_dir, mode='train', subjects=None, clip_length=75, data_part='all'):
        super().__init__()
        
        if subjects is None:
            subjects = [1, 2, 5, 7]

        self.mode = mode
        self.data_part = data_part
        self.subjects = subjects
        self.samples = []

        self.fmri_data_per_subject = []
        self.vae_latents_per_subject = []
        self.clip_latents_per_subject = []
        self.clip_emb_per_subject = []
        self.coco_labels_per_subject = []
        self.cnx_embeds_per_subject = []
        self.cnx_aug_embeds_per_subject = []
        self.imgs_per_subject = []

        print(f"--- Initializing Dataset in '{mode}' mode for subjects: {subjects} ---")
        print(f"--- Loading data part: '{self.data_part}' ---")

        for subj_num in tqdm(self.subjects, desc="Loading and filtering subject data"):
            subj_str = f"subj0{subj_num}"
            
            if mode == 'train':
                fmri_path = os.path.join('/home/yl/ssd_new/Dataset/fMRI/cvpr25_data/preprocessed_mri', f"subj0{subj_num}/train_fmri_avg_nsdgeneral.npy")
                cocoid_map_path = os.path.join('/home/yl/ssd_new/Dataset/fMRI/cvpr25_data/preprocessed_mri', f"subj0{subj_num}/train_fmri_avg_cocoid_nsdgeneral.npy")

                fmri_data_full = np.load(fmri_path)
                fmri_data_full=fmri_data_full.astype(np.float32)
                cocoids_full = np.load(cocoid_map_path)
                
                num_trials = len(fmri_data_full)
                half_point = num_trials // 2
                
                if self.data_part == 'first':
                    indices_slice = slice(0, half_point)
                elif self.data_part == 'second':
                    indices_slice = slice(half_point, num_trials)
                else: # 'all'
                    indices_slice = slice(None)

                self.fmri_data_per_subject.append(fmri_data_full[indices_slice])
                cocoids = cocoids_full[indices_slice]
            else: # test mode
                fmri_path = os.path.join(data_root_dir, '/home/yl/ssd_new/Dataset/fMRI/cvpr25_data/preprocessed_mri/', f"subj0{subj_num}/test_fmri_avg_nsdgeneral.npy")
                self.fmri_data_per_subject.append(np.load(fmri_path))
                cocoid_map_path = os.path.join('/home/yl/ssd_new/Dataset/fMRI/cvpr25_data/preprocessed_mri/', f"subj0{subj_num}/test_fmri_avg_cocoid_nsdgeneral.npy")
                cocoids = np.load(cocoid_map_path)
            
            cocoids_to_keep_set = set(str(c) for c in cocoids)

            def load_and_filter_pickle(path, keys_to_keep):
                with open(path, 'rb') as f:
                    full_dict = pickle.load(f)
                filtered_dict = {key: value for key, value in full_dict.items() if key in keys_to_keep}
                return filtered_dict

            vae_path = os.path.join(data_root_dir, 'target', 'imgVAE', subj_str, 'vae_latents.pkl')
            self.vae_latents_per_subject.append(load_and_filter_pickle(vae_path, cocoids_to_keep_set))

            clip_path = os.path.join(data_root_dir, 'target', 'textCLIP', subj_str, f'clipL[-1]_{clip_length}.pkl')
            self.clip_latents_per_subject.append(load_and_filter_pickle(clip_path, cocoids_to_keep_set))

            # clip_emb_path = os.path.join(data_root_dir, 'target', 'cap_latents', subj_str, 'clipGemb_75.pkl')
            # self.clip_emb_per_subject.append(load_and_filter_pickle(clip_emb_path, cocoids_to_keep_set))
            
            # label_path = os.path.join(data_root_dir, 'target', 'labels', subj_str, 'labels.pkl')
            # self.coco_labels_per_subject.append(load_and_filter_pickle(label_path, cocoids_to_keep_set))

            cnx_embed_path = os.path.join(data_root_dir, 'target', 'cnx_embeds', subj_str, 'cnx_embeds.pkl')
            self.cnx_embeds_per_subject.append(load_and_filter_pickle(cnx_embed_path, cocoids_to_keep_set))

            cnx_aug_path = os.path.join(data_root_dir, 'target', 'cnx_aug_embeds', subj_str, 'cnx_aug_embeds.pkl')
            self.cnx_aug_embeds_per_subject.append(load_and_filter_pickle(cnx_aug_path, cocoids_to_keep_set))

            img_path = os.path.join(data_root_dir, 'target', 'imgCLIP', subj_str, 'clip_bigG[-1]_embeddings.pkl')
            self.imgs_per_subject.append(load_and_filter_pickle(img_path, cocoids_to_keep_set))

            internal_subj_idx = len(self.fmri_data_per_subject) - 1
            for sample_idx_in_subj, cocoid in enumerate(cocoids):
                self.samples.append((subj_num, internal_subj_idx, sample_idx_in_subj, cocoid))
        
        print(f"--- Dataset initialization complete. Total samples: {len(self.samples)} ---")


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        subj_num, internal_subj_idx, sample_idx_in_subj, cocoid = self.samples[index]

        fmri = self.fmri_data_per_subject[internal_subj_idx][sample_idx_in_subj]
        str_cocoid = str(cocoid)
        
        vae_latent = self.vae_latents_per_subject[internal_subj_idx].get(str_cocoid)
        clip_latent = self.clip_latents_per_subject[internal_subj_idx].get(str_cocoid)
        # clip_emb = self.clip_emb_per_subject[internal_subj_idx].get(str_cocoid)
        # coco_label = self.coco_labels_per_subject[internal_subj_idx].get(str_cocoid)
        cnx_embed = self.cnx_embeds_per_subject[internal_subj_idx].get(str_cocoid)
        cnx_aug_embed = self.cnx_aug_embeds_per_subject[internal_subj_idx].get(str_cocoid)
        img_data = self.imgs_per_subject[internal_subj_idx].get(str_cocoid)

        return (
            torch.from_numpy(fmri.astype(np.float32)),
            torch.from_numpy(clip_latent.astype(np.float32)),
            internal_subj_idx,
            # torch.from_numpy(coco_label.astype(np.float32)),
            torch.from_numpy(vae_latent.astype(np.float32)),
            torch.from_numpy(cnx_embed.astype(np.float32)),
            torch.from_numpy(cnx_aug_embed.astype(np.float32)),
            torch.from_numpy(img_data.astype(np.float32))
            # torch.from_numpy(clip_emb.astype(np.float32))
        )

def custom_collate_fn(batch):
    """Collate the 7-item tuple returned by __getitem__."""
    (fmri_list, clip_list, subj_idx_list, 
     vae_list, cnx_norm_list, cnx_aug_list, img_list) = zip(*batch)

    clip_batch = torch.stack(clip_list, dim=0)
    # emb_batch = torch.stack(emb_list, dim=0)
    # label_batch = torch.stack(label_list, dim=0)
    vae_batch = torch.stack(vae_list, dim=0)
    cnx_norm_batch = torch.stack(cnx_norm_list, dim=0)
    cnx_aug_batch = torch.stack(cnx_aug_list, dim=0)
    img_batch = torch.stack(img_list, dim=0)

    subj_idx_batch = torch.tensor(subj_idx_list, dtype=torch.int64)

    return (
        fmri_list, 
        clip_batch, 
        subj_idx_batch, 
        # label_batch, 
        vae_batch,
        cnx_norm_batch,
        cnx_aug_batch,
        img_batch
        # emb_batch
    )

def get_dataloader(mode, batch_size, subjects=None, clip_length=75, num_workers=1, data_part='all'):
    data_root = '/home/yl/ssd_new/ymh/Datasets/NSD'
    
    dataset = NSDDataset(data_root, mode=mode, subjects=subjects, clip_length=clip_length, data_part=data_part)
    
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=(mode=='train'), 
        num_workers=num_workers,
        collate_fn=custom_collate_fn
    )
    
    print(f"Created '{mode}' DataLoader with {len(dataset)} samples.")
    return loader
