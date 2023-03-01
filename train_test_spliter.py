#%%
import re
from nltk.stem import WordNetLemmatizer
import pandas as pd
import numpy as np
from generate_wsdm_vg_data import RESULT_COLUMNS
from generate_wsdm_vqa_data import RESULT_COLUMNS as VQA_COLUMNS
#%
all_ = pd.read_csv("dataset/2023/vg_input/all+ofa+mplug.tsv", sep="\t", names=RESULT_COLUMNS)
#%%
def train_test_split(
    df,
    name: str,
    test_size: int=900,
    train_size: int=-1,
    with_aug: bool=False,
    random_state: int=42,
):
    import random
    random.seed(random_state)

    # specify the test size
    all_samp_idx = set(range(38990))
    if train_size == -1:
        train_size = 38990 - test_size

    # generate sample test data index
    test_samp_idx = np.array(
        random.sample(all_samp_idx, test_size)
    )
    # if with aug data, concat the aug test data index
    if with_aug:
        test_samp_idx = np.concatenate([
            test_samp_idx,
            test_samp_idx + 38990,
        ])
    # get the test data
    test_samp = df.iloc[test_samp_idx]

    # get the train data index
    train_samp_idx = np.array(
        list(all_samp_idx.difference(set(test_samp_idx))),
        dtype=int,
    )
    # if train size is not all, sample the train data
    train_samp_idx = np.array(
        random.sample(set(train_samp_idx), train_size)
    )
    # if with aug data, concat the aug train data index
    if with_aug:
        train_samp_idx = np.concatenate([
            train_samp_idx,
            train_samp_idx + 38990,
        ])
    # get the train data
    train_samp: pd.DataFrame = df.iloc[train_samp_idx]

    # save the data
    if with_aug:
        train_size *= 2
        test_size *= 2
    train_samp.to_csv(
        f"dataset/2023/vg_input/train_test_split/{name}_train{train_size}.tsv",
        sep="\t",
        index=False,
        header=False,
    )
    test_samp.to_csv(
        f"dataset/2023/vg_input/train_test_split/{name}_test{test_size}.tsv",
        sep="\t",
        index=False,
        header=False,
    )

    return train_samp, test_samp
# %%
mplug = pd.read_csv("dataset/2023/vg_input/all+mplug.tsv", sep="\t", names=RESULT_COLUMNS)
# %%
train, test = train_test_split(
    df=all_,
    name='all+mplug',
    test_size=1000,
    train_size=-1,
    with_aug=True,
)
# %%
