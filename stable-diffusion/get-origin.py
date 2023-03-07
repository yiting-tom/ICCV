#%%
import cv2
from typing import List, Optional, Union, Tuple

import numpy as np
import torch
import pandas as pd
import PIL
import matplotlib.pyplot as plt

from utils import img_formatter

#%%
RESULT_COLUMNS = [
    'unique_id', 'image_id', 'text', 'bbox', 'image'
]
df = pd.read_pickle('tmp_train.pkl')
#%%
plt.imshow(pix)
#%%
def save_box_img(data, save_path, prompt):
    start, end = np.array(eval(data['bbox'])).reshape(2, 2)
    img = np.array(img_formatter.base64_to_np(data['image']))
    pix = cv2.rectangle(img, start, end, (0, 255, 0), 1)
    pix = cv2.resize(pix, (512, 512))
    plt.title(prompt)
    plt.imsave(save_path, pix)

#%%
from tqdm import tqdm

for i, row in tqdm(df.iterrows(), total=len(df)):
    prompt = row['text']
    bbox = row['bbox']
    iid = row['image_id']
    filepath = f"./diff-ori/{iid:012d}.jpg"
    save_box_img(row, filepath, prompt)
#%%
def cat_aug_ori(data, save_path, prompt):
    prompt = data['text']
    bbox = data['bbox']
    iid = data['image_id']
    ori = PIL.Image.open(f"./diff-ori/{iid:012d}.jpg")
    aug = PIL.Image.open(f"./diff-aug/{iid:012d}-0.jpg")
    filepath = f"./diff-cat/{iid:012d}.jpg"
#%%
def save_cat(data):
    prompt = data['text']
    bbox = data['bbox']
    iid = data['image_id']
    print(iid, prompt)
    ori = PIL.Image.open(f"./diff-ori/{iid:012d}.jpg")
    aug = PIL.Image.open(f"./diff-aug/{iid:012d}-0.jpg")
    filepath = f"./diff-cat/{iid:012d}.jpg"

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].imshow(ori)
    ax[0].set_title('original')
    ax[1].imshow(aug)
    ax[1].set_title(f'augmented with prompt: {prompt}')
    fig.tight_layout()
    # disable all ticks
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    fig.savefig(filepath)
# %%
df = df.iloc[:30]
#%%
for i, row in df.iterrows():
    _ = save_cat(row)