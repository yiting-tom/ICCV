import re
import unicodedata
from typing import Union
from pathlib import Path

import pandas as pd

from configs import paths


# full-width to half-width and translate to ascii with unicodedata
def full_width_to_half_width(s: str) -> str:
	return unicodedata.normalize('NFKD', s) \
		.encode('ascii', 'ignore') \
		.decode('utf-8', 'ignore')

def translate_head_punctuations(s: str) -> str:
	sp = s.split(' ')
	x = sp[0]
	x = re.sub(r'w[wh]h', 'wh', x)
	x = re.sub(r'\'s', ' is', x)
	x = re.sub(r'\'re', ' are', x)
	x = re.sub(r'\'|\"|-|_|\,|/|\.', '', x)
	sp[0] = x
	result = ' '.join(sp).lower()
	if not result.endswith('?'):
		result += '?'
	return result.lower()

def reformat_question(q: str) -> str:
	q = q.lower()
	q = full_width_to_half_width(q)
	q = translate_head_punctuations(q)
	return q

def series_reformat_question(q: pd.Series) -> pd.Series:
    return q.map(reformat_question)


def url_to_img_filename(url: str) -> str:
    return url.split('/')[-1]

def url_to_img_id(url: str) -> str:
    return url_to_img_filename(url).split('.')[0]

def url_to_img_filepath(url: str, parent_dir: Union[str, Path]) -> str:
	return str(parent_dir / url.split('/')[-1])

def wsdmdata_to_url(data: dict) -> str:
    return f"https://toloka-cdn.azureedge.net/wsdmcup2023/{data['img_id']}.jpg"

def id_to_url(id: str) -> str:
    return f"https://toloka-cdn.azureedge.net/wsdmcup2023/{id}.jpg"

def id_to_img_filepath(id: str, stage: str = 'train') -> str:
    return str(paths.WSDM / stage / (id + '.jpg'))

def series_url_to_img_filename(url: pd.Series) -> pd.Series:
	return url.map(url_to_img_filename)

def series_url_to_img_filepath(url: pd.Series) -> pd.Series:
	return url.map(url_to_img_filepath)