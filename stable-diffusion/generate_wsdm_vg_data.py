#%%
"""
The information are separated by tabs.
0. uniq-id
1. image-id
2. text
3. region-coord (separated by commas)
4. image base64 string

e.g.)
79_1    237367  A woman in a white blouse holding a glass of wine.  230.79,121.75,423.66,463.06 9j/4AAQ...1pAz/9k=

"""
import os
import logging
import argparse
from pathlib import Path
import subprocess

import pandas as pd

from configs import paths
from generate_wsdm_vqa_data import RESULT_COLUMNS as VQA_COLUMNS
from generate_wsdm_vqa_data import VQA_DATASET_PATH_FMT


L: logging.Logger = logging.getLogger(logging.basicConfig(level=logging.INFO))
VG_DATASET_PATH_FMT = str(paths.ROOT / "dataset" / "wsdm_vg_data" / "wsdm_vg_{split}.tsv")
VQA_ANSWER_DIR_FMT = str(paths.ROOT / 'results' / 'wsdm_vqa_{split}_zeroshot_b{beam}')
RESULT_COLUMNS = [
    'unique_id', 'image_id', 'text', 'bbox', 'image'
]

#%%
def load_wsdm_vqa_data(split: str):
    filepath = VQA_DATASET_PATH_FMT.format(split=split)
    try:
        L.info(f"Loading {filepath}")
        return pd.read_csv(
            filepath,
            sep='\t',
            names=VQA_COLUMNS
        )
    except FileNotFoundError as e:
        L.info(e)
        os.system(f"cd {paths.ROOT} && python3 generate_wsdm_vqa_data.py")

def load_wsdm_vqa_data_answer(split: str, beam: int = 30):
    dataset_path = VQA_DATASET_PATH_FMT.format(split=split)
    result_path = VQA_ANSWER_DIR_FMT.format(split=split, beam=beam)
    model_path = paths.ROOT / "checkpoints" / "ofa_huge.pt"
    result_jsonfile = Path(result_path) / f"{split}_predict.json"
    try:
        L.info(f"Loading {result_jsonfile}")
        return pd.read_json(result_jsonfile)
    except FileNotFoundError as e:
        L.info(e)
        cmd = (
            f"cd {paths.ROOT}/run_scripts/vqa"
            " &&"
            " sh evaluate_wsdm_vqa_zeroshot.sh"
            f" {split}"          # split
            f" {beam}"           # beam
            f" {dataset_path}"   # dataset_path
            f" {result_path}"    # result_path
            f" {model_path}"     # model_path
        )
        L.info(f"Running: {cmd}")
        p = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
        )
        return_code = p.wait()
        L.info(f"Return code: {return_code}")
        L.info(f"Loading {result_jsonfile}")
        return pd.read_json(result_jsonfile)

def process_static_data(
    split: str,
    df: pd.DataFrame,
    df_vqa: pd.DataFrame,
    df_vqa_answer: pd.DataFrame,
) -> pd.DataFrame:
    # pack bbox
    if 'test' not in split:
        df_vqa['bbox'] = df[['left', 'top', 'right', 'bottom']]\
            .astype(str)\
            .agg(','.join, axis=1)
    else:
        df_vqa['bbox'] = df[['height', 'width']]\
            .astype(str)\
            .agg(
                lambda x: ','.join(
                    ('0', x['width'], '0', x['height'])
                ),
                axis=1,
            )

    # rename question_id as unique_id
    df_vqa.rename(
        columns={'question_id': 'unique_id'},
        inplace=True,
    )
    # merge vqa answer
    df_vg = df_vqa.join(
        df_vqa_answer,
        on='unique_id',
        rsuffix='_text',
    )
    # rename answer as text
    df_vg['text'] = df_vg['answer_text']
    return df_vg


#%%
def main(split='train'):
    tsv_filepath = VG_DATASET_PATH_FMT.format(split=split)
    if tsv_filepath.exists():
        L.info(f"{tsv_filepath} already exists. Skip.")
        return None
    # load vqa data
    L.info(f"Loading vqa data: {split}")
    df_vqa = load_wsdm_vqa_data(split)
    # load original csv provided by official
    L.info(f"Loading original csv data: {split}")
    df = pd.read_csv(paths.ORIGINAL / "csvs" / f"{split}.csv")
    # load vqa answer generated by zeroshot
    L.info(f"Loading vqa answer data: {split}")
    df_vqa_answer = load_wsdm_vqa_data_answer(split)
    # process static data
    L.info(f"Processing static data: {split}")
    df_vg = process_static_data(
        split=split,
        df=df,
        df_vqa=df_vqa,
        df_vqa_answer=df_vqa_answer,
    )
    # only keep necessary columns
    L.info(f"Saving VG data: {split} to {tsv_filepath}")
    df_vg[RESULT_COLUMNS].to_csv(
        tsv_filepath,
        sep='\t',
        index=False,
        header=False,
    )

    L.info(f"Saving VG data: {split} to {tsv_filepath}")
    df_vg[RESULT_COLUMNS].to_csv(
        tsv_filepath,
        sep='\t',
        index=False,
        header=False,
    )

if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument("--split", type=str, default="train", choices=["train", "test_public", "train_sample"])
    a.add_argument("--beam", type=int, default=30)
    a = a.parse_args()
    main(a.split)