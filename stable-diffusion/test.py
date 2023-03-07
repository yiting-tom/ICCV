#%%
from pathlib import Path
from typing import Tuple
import cv2
import numpy as np
import torch
import pandas as pd
import PIL

from diffusers import StableDiffusionInpaintPipeline

from utils import img_formatter

# %%
device = "cuda:0"

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16,
    revision="fp16",
    safety_checker=None,
).to(device)

#%%
RESULT_COLUMNS = [
    'unique_id', 'image_id', 'text', 'bbox', 'image'
]
df = pd.read_csv(
    "/home/P76104419/ICCV/dataset/base64/train-fus.csv",
    sep="\t",
    names=RESULT_COLUMNS,
)

def process_img_mask(image: PIL.Image, bbox: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """process_img_mask 

    scale image to 512x512, and mask the bbox area

    Args:
        image (PIL.Image): The image to be processed
        bbox (np.array): The bbox to be masked

    Returns:
        tuple: The processed image and mask
    """
    pix = np.array(image)
    # scale image to 512x512x3
    pix = cv2.resize(pix, (512, 512))
    # mask the bbox area
    mask = np.zeros(pix.shape, dtype=np.uint8)
    assert mask.shape == pix.shape == (512, 512, 3), f"mask.shape: {mask.shape}, pix.shape: {pix.shape}"
    # scale the bbox to 512x512
    bbox[0], bbox[2] = bbox[0] * 512 / image.width, bbox[2] * 512 / image.width
    bbox[1], bbox[3] = bbox[1] * 512 / image.height, bbox[3] * 512 / image.height
    mask[bbox[1]:bbox[3], bbox[0]:bbox[2], :] = 255
    # convert to PIL image
    img = PIL.Image.fromarray(pix)
    mask = PIL.Image.fromarray(mask)
    return img, mask

#%%
from tqdm import tqdm
for i, row in tqdm(df.iterrows(), total=len(df)):
    prompt = row['text']
    bbox = row['bbox']
    iid = row['image_id']
    image = img_formatter.base64_to_pil_image(row['image'])
    img, mask = process_img_mask(image, bbox=np.array(eval(bbox)))

    guidance_scale=2.1
    num_samples = 3
    generator = torch.Generator(device=device).manual_seed(152)

    images = pipe(
        prompt=prompt,
        image=img,
        mask_image=mask,
        guidance_scale=guidance_scale,
        generator=generator,
        num_images_per_prompt=num_samples,
    ).images

    for k, out in enumerate(images):
        filepath = Path(f"./diff-aug/{iid:012d}-{k}.jpg")
        filepath.parent.mkdir(exist_ok=True)
        out.save(filepath)
        print(f"save {filepath}")
