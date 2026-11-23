from __future__ import annotations
from pathlib import Path
import pandas as pd
from .preprocessing import ID_COLUMN, TARGET
from .validation import stratified_split

def find_repo_root(start: Path | None = None) -> Path:
    here = (start or Path.cwd()).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "data" / "train.csv").exists():
            return candidate
    raise FileNotFoundError("Could not find data/train.csv from " + str(here))

ROOT = find_repo_root(Path(__file__).parent)
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"

def load_competition_data():
    train = pd.read_csv(DATA_DIR / "train.csv", dtype={ID_COLUMN: str})
    test = pd.read_csv(DATA_DIR / "test.csv", dtype={ID_COLUMN: str})
    return train, test

def development_split(train_df, test_size=0.2, random_state=42):
    train_idx, holdout_idx = stratified_split(train_df[TARGET].to_numpy(), test_size, random_state)
    return train_df.iloc[train_idx].reset_index(drop=True), train_df.iloc[holdout_idx].reset_index(drop=True)