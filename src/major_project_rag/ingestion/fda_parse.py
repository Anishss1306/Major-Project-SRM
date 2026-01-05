from __future__ import annotations

import glob
from pathlib import Path
from typing import Optional

import pandas as pd

from major_project_rag.config import PATHS


def _find_case_insensitive_file(directory: Path, filename: str) -> Optional[Path]:
    files = [Path(p) for p in glob.glob(str(Path(directory) / "*"))]
    for f in files:
        if filename.lower() in f.name.lower():
            return f
    return None


def _load_fda_table(raw_fda_dir: Path, filename: str) -> pd.DataFrame:
    target = _find_case_insensitive_file(raw_fda_dir, filename)
    if not target:
        raise FileNotFoundError(f"Could not find {filename} in {raw_fda_dir}")

    print(f"📖 Loading {target.name}...")
    try:
        return pd.read_csv(target, sep="\t", encoding="cp1252", on_bad_lines="skip")
    except Exception:
        return pd.read_csv(target, sep="\t", encoding="utf-8", on_bad_lines="skip")


def parse_fda_drugs(
    *,
    raw_fda_dir: Path = PATHS.raw_fda,
    output_file: Path = PATHS.fda_parquet,
) -> Path:
    raw_fda_dir = Path(raw_fda_dir)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    products = _load_fda_table(raw_fda_dir, "Products.txt")
    applications = _load_fda_table(raw_fda_dir, "Applications.txt")

    products = products.rename(
        columns={
            "DrugName": "drug_name",
            "ActiveIngredient": "active_ingredient",
            "ApplNo": "appl_no",
            "Form": "form",
            "Strength": "strength",
        }
    )
    applications = applications.rename(columns={"ApplNo": "appl_no", "SponsorName": "sponsor_name"})

    print("🔄 Merging Product and Application data...")
    merged = pd.merge(products, applications[["appl_no", "sponsor_name"]], on="appl_no", how="left")
    merged = merged.fillna("Unknown")

    # Normalize to string columns to avoid PyArrow dtype coercion issues.
    print("🛠️  Normalizing data types...")
    for col in merged.columns:
        merged[col] = merged[col].astype(str)

    merged["drug_name"] = merged["drug_name"].str.title()
    merged["active_ingredient"] = merged["active_ingredient"].str.title()

    print(f"💾 Saving {len(merged)} drug records to {output_file}...")
    merged.to_parquet(output_file, index=False)
    print("✅ FDA parsing complete.")
    return output_file


def main() -> int:
    parse_fda_drugs()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


