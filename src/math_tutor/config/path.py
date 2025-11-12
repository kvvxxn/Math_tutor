from pathlib import Path
import os

# rc/math_tutor/config/path.py 기준
ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data"
ORIGINAL_DATA_DIR = ROOT_DIR / "original_data"
DOCUMENT_DIR = ROOT_DIR / "documents"
VECTORDB_DIR = ROOT_DIR / "vectordb"
CONFIG_DIR =  Path(__file__).resolve().parent