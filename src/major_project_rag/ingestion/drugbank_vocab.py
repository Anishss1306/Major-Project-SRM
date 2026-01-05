from __future__ import annotations

import glob
from pathlib import Path
from typing import Optional

import pandas as pd

from major_project_rag.config import PATHS


def find_drugbank_vocab_csv(raw_dir: Path, pattern: str = "*vocab*.csv") -> Optional[Path]:
    matches = [Path(p) for p in glob.glob(str(Path(raw_dir) / pattern))]
    return matches[0] if matches else None


def parse_drugbank_vocab(
    *,
    raw_dir: Path = PATHS.raw,
    output_file: Path = PATHS.drugbank_vocab_parquet,
) -> Path:
    raw_dir = Path(raw_dir)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    vocab_csv = find_drugbank_vocab_csv(raw_dir)
    if not vocab_csv:
        raise FileNotFoundError(
            f"No DrugBank vocabulary CSV found under {raw_dir} (pattern '*vocab*.csv'). "
            "Place 'drugbank vocabulary.csv' (or similar) in data/raw/."
        )

    print(f"📖 Loading {vocab_csv.name}...")
    df = pd.read_csv(vocab_csv)

    required_cols = ["Common name", "Synonyms", "DrugBank ID"]
    if "Common name" not in df.columns:
        raise ValueError(f"'Common name' column not found. Columns present: {df.columns.tolist()}")

    available_cols = [c for c in required_cols if c in df.columns]
    df = df[available_cols].copy()

    df = df.rename(
        columns={
            "Common name": "drug_name",
            "Synonyms": "synonyms",
            "DrugBank ID": "drugbank_id",
        }
    )
    if "synonyms" in df.columns:
        df["synonyms"] = df["synonyms"].fillna("")

    print(f"💾 Saving {len(df)} vocabulary records to {output_file}...")
    df.to_parquet(output_file, index=False)
    print("✅ DrugBank vocabulary parsing complete.")
    return output_file


def main() -> int:
    parse_drugbank_vocab()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


