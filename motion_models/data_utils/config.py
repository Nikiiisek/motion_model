from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

DATASET_ROOT = Path(os.getenv("DATASET_ROOT"))
PROCESSED_DIR = DATASET_ROOT / "processed_16f"