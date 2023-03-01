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