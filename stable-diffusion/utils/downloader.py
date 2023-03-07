from typing import Optional
from joblib import Parallel, delayed
from tqdm.auto import tqdm
import pandas as pd
import wget
from pathlib import Path

def download_images(
    df: pd.DataFrame,
    image_dir: Path,
    image_col_name: Optional[str] = 'image',
) -> None:
    """download_images

    Args:
        df (pd.DataFrame): The dataframe containing the image urls.
        image_dir (Path): The directory to save the images.
        image_col_name (Optional[str], optional): The name of column to download image by url. Defaults to 'image'.
    """
    image_dir.mkdir(parents=True, exist_ok=True)

    _ = Parallel(n_jobs=100)(
        delayed(wget.download)(img_url, out=str(image_dir))
        for img_url in tqdm(df[image_col_name])
    )