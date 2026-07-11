import argparse
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from qwen_vl_utils import process_vision_info
import pandas as pd
from tqdm import tqdm
from PIL import Image
import torch
import time
import numpy as np
from torch.utils.data import DataLoader, Dataset
import json

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor


DEFAULT_NSD_SUBJECTS = ("01", "02", "05", "07")
DEFAULT_CAPTION_LENGTHS = (30, 45, 60, 75)


def parse_subjects(subjects):
    parsed_subjects = []
    for subject in subjects:
        for part in str(subject).split(","):
            subject_id = part.strip()
            if subject_id.lower().startswith("subj"):
                subject_id = subject_id[4:]
            if not subject_id.isdigit():
                raise ValueError(f"Invalid subject id: {subject}")
            parsed_subjects.append(subject_id.zfill(2))

    return parsed_subjects


def parse_caption_lengths(lengths):
    parsed_lengths = []
    for length in lengths:
        for part in str(length).split(","):
            length_value = part.strip()
            if not length_value.isdigit() or int(length_value) <= 0:
                raise ValueError(f"Invalid caption length: {length}")
            parsed_lengths.append(int(length_value))

    return parsed_lengths


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate MimeVis semantic captions for NSD subjects."
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=list(DEFAULT_NSD_SUBJECTS),
        help="NSD subject ids to process. Defaults to the paper subjects: 01 02 05 07.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for Qwen2-VL caption generation.",
    )
    parser.add_argument(
        "--caption_lengths",
        nargs="+",
        default=list(DEFAULT_CAPTION_LENGTHS),
        help="MimeVis caption length limits. Defaults to the paper levels: 30 45 60 75.",
    )
    return parser.parse_args()



class BatchDataset(Dataset):
    def __init__(self, cocoids):
        self.cocoids = cocoids

    def __len__(self):
        return len(self.cocoids)

    def __getitem__(self, idx):
        return self.cocoids[idx]

class Qwen2VL:
    def __init__(self, model_path = None, max_new_tokens = 1024, min_pixels = 256*28*28, max_pixels = 1280*28*28):
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.gen_config = {
            "max_new_tokens": 512,
        }
    
    def parse_input(self, query=None, imgs=None, history = None, **kwargs):
        if (history is None and imgs is None) or (history is not None and imgs is not None):
            raise ValueError("Exactly one of query or history must be provided.")
    
        if history is not None:
            messages = history
        else:
            messages = []

        if imgs is not None:
            for img in imgs:
                msg = [{"role": "user", "content": [{"type": "image", "image": img},
                                                {"type": "text", "text": query}]}]
                messages.append(msg)
        else:
            messages = [messages[i] + [{"role": "user", "content": query.replace("***", kwargs['captions'][i])}] for i in range(len(messages))]
        
        return messages
        
  



    def chat(self, query = None, imgs = None, history = None, **kwargs):

        messages = self.parse_input(query, imgs, history, **kwargs)

        text = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, add_vision_id=True) for msg in messages]
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )


        inputs = inputs.to("cuda")
        generated_ids = self.model.generate(**inputs, **self.gen_config)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        if history is None:
            history = messages

        history = [history[i] + [{"role": "assistant", "content": response[i]}] for i in range(len(messages))]


        del inputs, generated_ids, generated_ids_trimmed
        torch.cuda.empty_cache()
        
        return response, history


def normalize_coco_caption(caption):
    if isinstance(caption, list):
        return caption[0]
    return str(caption)


def save_caption_csv(caption_save_pth, processed_cocoids, coco_caption_texts, fine_captions, caption_length, legacy_save_pth=None):
    data = {
        'cocoid': processed_cocoids,
        'coco_cap': coco_caption_texts,
        f'v{caption_length}': fine_captions,
        'fine_cap': fine_captions,
    }

    df = pd.DataFrame(data)
    df.to_csv(caption_save_pth)
    if legacy_save_pth is not None:
        df.to_csv(legacy_save_pth)


def batch_call_with_local_file(dataloader, model, prompts, img_dir, caption_save_pth, caption_length, legacy_save_pth=None, generate_img = False):
    coco_captions = json.load(open('/home/yl/ssd_new/Dataset/fMRI/cvpr25_data/coco_captions.json', 'r'))
    fine_captions = []
    processed_cocoids = []
    coco_caption_texts = []
    since = time.time()

    for i, cocoids in tqdm(enumerate(dataloader), total=len(dataloader)):

        response, history = model.chat(query=prompts[0], imgs = [f"{img_dir}/{str(cocoid.item())}.jpg" for cocoid in cocoids])
        batch_coco_captions = [normalize_coco_caption(coco_captions[str(cocoid.item())]) for cocoid in cocoids]
        for prompt in prompts[1:]:
            response, history = model.chat(query=prompt, history=history, captions=batch_coco_captions)

            # if generate_img:
            #     import caption2img
            #     import utils
            #     for cocoid, resp in zip(cocoids, response):
            #         img = caption2img.caption2img(resp)
            #         new_img = utils.img_compare([cocoid.item()], img, img_dir)
            #         new_img[0].save(os.path.join(str(cocoid.item())+'.jpg'))
            #         a = 1
            
        processed_cocoids.extend(cocoids.cpu().tolist())
        coco_caption_texts.extend(batch_coco_captions)
        fine_captions.extend(response)
        cost_time_min = (time.time() - since ) // 60
        cost_time_sec = (time.time() - since ) % 60
        print(f'{cost_time_min} min {cost_time_sec} sec')
            
        if len(fine_captions) % 100 == 0:
            save_caption_csv(caption_save_pth, processed_cocoids, coco_caption_texts, fine_captions, caption_length, legacy_save_pth)
    
    save_caption_csv(caption_save_pth, processed_cocoids, coco_caption_texts, fine_captions, caption_length, legacy_save_pth)


if __name__ == '__main__':
    args = parse_args()
    subjects = parse_subjects(args.subjects)
    caption_lengths = parse_caption_lengths(args.caption_lengths)
    first_round_prompt = "What visual semantics do humans perceive from this visual stimulus in 3 seconds?"
    
    for subj in subjects:
        print('-'*10, 'Processing subj {}'.format(subj))
        img_dir = '/home/yl/ssd_new/Dataset/fMRI/cvpr25_data/preprocessed_img/subj{}'.format(subj)
        subj_info = pd.read_csv(os.path.join('/home/yl/ssd_new/Dataset/fMRI/cvpr25_data/', 'subj{}_data_info.csv'.format(subj)))
        valid_info_num = len(np.load('/home/yl/ssd_new/Dataset/fMRI/cvpr25_data/roi_data/subj{}/{}.npy'.format(subj, 'nsdgeneral'))) # last three session are not public 
        cocoids = subj_info['cocoid'][:valid_info_num].tolist()
        cocoids = list(set(cocoids))
        cocoids.sort()
        
        dataset = BatchDataset(cocoids)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size)

        model = Qwen2VL("/home/yl/ssd_new/pretrain_weights/qwen2_vl_2b_instruct")

        # accelerator = accelerate.Accelerator()

        # model, dataloader = accelerator.prepare(model, dataloader)

        save_dir = '/home/yl/ssd_new/Dataset/fMRI/cvpr25_data/subj_target/subj{}/'.format(subj)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
       
        for caption_length in caption_lengths:
            prompts = [
                first_round_prompt,
                f"""Expand the basic caption *** into a detailed one using the previously discussed semantics. Aim for clarity and precision in your description, suitable for a text-to-image model input, and keep it under {caption_length} words""",
            ]
            caption_save_pth = '/home/yl/ssd_new/Dataset/fMRI/cvpr25_data/subj_target/subj{}/qwen2b_2runs_1cap_v{}.csv'.format(subj, caption_length)
            legacy_save_pth = None
            if caption_length == 75:
                legacy_save_pth = '/home/yl/ssd_new/Dataset/fMRI/cvpr25_data/subj_target/subj{}/qwen2b_2runs_1cap_ext.csv'.format(subj)
            batch_call_with_local_file(dataloader, model, prompts, img_dir, caption_save_pth, caption_length, legacy_save_pth, generate_img = True)
