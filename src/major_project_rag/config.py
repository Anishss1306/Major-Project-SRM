from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def project_root() -> Path:
    # .../Major_Project_RAG/src/major_project_rag/config.py -> parents[2] == repo root
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    root: Path = project_root()

    data: Path = root / "data"
    raw: Path = data / "raw"
    processed: Path = data / "processed"

    raw_pubmed: Path = raw / "pubmed"
    raw_fda: Path = raw / "fda"

    pubmed_parquet: Path = processed / "pubmed_extracted.parquet"
    drugbank_vocab_parquet: Path = processed / "drugbank_vocab.parquet"
    fda_parquet: Path = processed / "fda_drugs.parquet"

    chroma_db_dir: Path = data / "chroma_db"


PATHS = Paths()


