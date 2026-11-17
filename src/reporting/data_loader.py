
import json
from pathlib import Path

import pandas as pd

from src.reporting.config import RES_DIR


def load_all_results():

    records = []

    if not RES_DIR.exists():
        print(f"Results directory not found: {RES_DIR}")
        return pd.DataFrame()

    for file_path in RES_DIR.glob("*.json"):

        try:
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)

            data["filename"] = file_path.name
            records.append(data)

        except (json.JSONDecodeError, OSError) as error:
            print(f"Could not read {file_path.name}: {error}")

    df = pd.DataFrame(records)

    if df.empty:
        return df

    # Compatibility with older result files

    if "augmentation" not in df.columns:
        df["augmentation"] = False
    else:
        df["augmentation"] = df["augmentation"].fillna(False)

    if "pretrained" not in df.columns:
        df["pretrained"] = False

    if "no_ae" not in df.columns:
        df["no_ae"] = False

    return df
