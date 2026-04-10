from diffusers import StableDiffusionImg2ImgPipeline
import torch
from PIL import Image
import numpy as np
import os
from pytorch_lightning import seed_everything
seed_everything(42)
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "/home/yl/ssd_new/pretrain_weights/stable-diffusion-v1-4", torch_dtype=torch.float16
).to("cuda")
# label_mineye = np.load("/home/ymh/metric/ReconImages/Mindeye2/shared1000_cocoids.npy")
label = np.load("/home/ymh/pami25/codes/recon/shared1000_cocoids.npy")
cocoid_to_our = {coco_id: idx for idx, coco_id in enumerate(label)}
if True:
    for subj in [1,2,5,7]:
        if subj == 1:
            black =['88835', '416279', '433021', '164502', '408922', '362677', '277778', '161062', '320533', '516634', '59321', '416972', '189888', '61174', '116603', '536786', '200391', '238201']
        elif subj == 2:
            black = ['24287', '265745', '139344', '453221', '332498', '546140', '323970', '274978', '541258', '200391']
        elif subj == 5:
            black = ['88835', '290192', '110611', '67961', '362677', '435902', '139561', '140590', '43829', '59321', '416972', '37779', '469816', '40685', '340734', '200391']
        elif subj == 7:
            black = ['531515', '334048', '310870', '293372', '96161', '290192', '140590', '37358', '144193', '7567', '320533', '329641', '516634', '266622', '61174', '376407']
        pred = torch.load(f"/home/yl/ssd_new/ymh/pami25/weights/vi/test_subj0{subj}_10e.pth")
        prior_out = pred["text_results"]
        for cocoid_str in black:
            cocoid = int(cocoid_str)
            if cocoid not in cocoid_to_our:
                print(f"COCO ID {cocoid} not found in our dataset")
                continue
            our_idx = cocoid_to_our[cocoid]

            init_image = Image.open(f"/home/yl/ssd_new/ymh/pami25/results/subj0{subj}_pred/I2I_E_vi/{cocoid}.jpg").convert("RGB").resize((512, 512))
            carr = prior_out[our_idx, :].reshape(77, 768)
            c = torch.Tensor(carr).unsqueeze(0).to("cuda")
            result = pipe(prompt_embeds=c, 
                        negative_prompt="low quality, bad quality, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, deformed, poorly drawn face, deformed face, deformed hands, deformed fingers, deformed digits, deformed body,",
                        image=init_image, strength=0.75, guidance_scale=7.5).images[0]
            result.save(f"/home/yl/ssd_new/ymh/pami25/results/subj0{subj}_pred/TextEhenced_0.5/{cocoid}.jpg")
if False:
    for subj in [1,2,5,7]:
        pred = torch.load(f"/home/yl/ssd_new/ymh/pami25/weights/vi/test_subj0{subj}_10e.pth")
        prior_out = pred["text_results"]
        for id in range(0,982):
            cocoid = label[id]
            if True:
                init_image = Image.open(f"/home/yl/ssd_new/ymh/pami25/results/subj0{subj}_pred/I2I_E_vi/{cocoid}.jpg").convert("RGB").resize((512, 512))
                carr = prior_out[id, :].reshape(77, 768)
                c = carr.unsqueeze(0).to("cuda")
                result = pipe(prompt_embeds=c, 
                            negative_prompt="low quality, bad quality, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, deformed, poorly drawn face, deformed face, deformed hands, deformed fingers, deformed digits, deformed body,",
                            image=init_image, strength=0.75, guidance_scale=7.5, num_inference_steps = 50).images[0]
                os.makedirs(f"/home/yl/ssd_new/ymh/pami25/results/subj0{subj}_pred/TextEhenced_0.5", exist_ok=True)
                result.save(f"/home/yl/ssd_new/ymh/pami25/results/subj0{subj}_pred/TextEhenced_0.5/{cocoid}.jpg")