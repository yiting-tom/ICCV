from pathlib import Path

ROOT: Path = Path(__file__).parent.parent

# Change "reproduction" to "/mnt" for reproducing the results
DATASET: Path = ROOT / "dataset"
ORIGINAL: Path = DATASET / "original"

# The original data provided by WSDM
TRAIN_CSV: Path = ORIGINAL / "train.csv"
TRAIN_SAMPLE_CSV: Path = ORIGINAL / "train_sample.csv"
TEST_PUBLIC_CSV: Path = ORIGINAL / "test_public.csv"
TEST_PRIVATE_CSV: Path = ORIGINAL / "test.csv"

# The generated data
GENERATED: Path = DATASET / "generated"
VG_DATASET: Path = GENERATED / "vg_data"
VQA_DATASET: Path = GENERATED / "vqa_data"

# Change "reproduction" to "/mnt" for reproducing the results
DATASET: Path = ROOT / "dataset"
OFFICIAL: Path = DATASET / "official"
OFFICIAL_IMG: Path = OFFICIAL / "images"

BASE64_DIR: Path = DATASET / "base64"
VG_64: Path = BASE64_DIR / "vg"
VQA_64: Path = BASE64_DIR / "vqa"

HINT_DIR: Path = DATASET / "hints"
OFA_HINT: Path = HINT_DIR / "ofa"
MPLUG_HINT: Path = HINT_DIR / "mplug"

# VQA SOTA models
VQA_OFA_MODEL_PATH = ROOT / "backbone" / "ofa_huge.pt"
VQA_OFA_RESULT_DIR = ROOT / "dataset" / "ofa-result"
VQA_MPLUG_RESULT_DIR = ROOT / "dataset" / "mplug-result"