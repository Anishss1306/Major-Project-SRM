from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

from major_project_rag.config import PATHS
from major_project_rag.ingestion.chunking import chunk_text


@dataclass(frozen=True)
class ChromaIndexConfig:
    parquet_path: Path = PATHS.pubmed_parquet
    chroma_dir: Path = PATHS.chroma_db_dir
    collection_name: str = "pubmed_evidence"
    embedding_model: str = "all-MiniLM-L6-v2"
    batch_size: int = 100
    min_abstract_len: int = 20
    chunk_size: int = 800
    chunk_overlap: int = 100


def build_or_update_index(
    *,
    cfg: ChromaIndexConfig = ChromaIndexConfig(),
    rebuild: bool = False,
    limit: Optional[int] = None,
) -> None:
    import chromadb
    from sentence_transformers import SentenceTransformer

    parquet_path = Path(cfg.parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Processed parquet not found: {parquet_path}")

    print(f"Loading data from {parquet_path}...")
    df = pd.read_parquet(parquet_path)

    # Filter out empty / very short abstracts
    df = df[df["abstract"].astype(str).str.len() > cfg.min_abstract_len]
    if limit is not None:
        df = df.head(int(limit))
    print(f"Total articles to index: {len(df)}")

    print(f"Loading embedding model ({cfg.embedding_model})...")
    model = SentenceTransformer(cfg.embedding_model)

    print(f"Initializing ChromaDB at {cfg.chroma_dir}...")
    client = chromadb.PersistentClient(path=str(cfg.chroma_dir))

    if rebuild:
        try:
            client.delete_collection(cfg.collection_name)
            print("Deleted existing collection to rebuild.")
        except Exception:
            print("Collection not found; building a fresh one.")
        collection = client.create_collection(name=cfg.collection_name)
    else:
        collection = client.get_or_create_collection(name=cfg.collection_name)

    records = df.to_dict("records")
    print("Starting chunking -> embedding -> upsert...")

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []

    def flush() -> None:
        if not documents:
            return
        embeddings = model.encode(documents).tolist()
        collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
        ids.clear()
        documents.clear()
        metadatas.clear()

    for record_idx, r in enumerate(tqdm(records, desc="Chunking + indexing")):
        pmid = r.get("pmid")
        base_id = str(pmid) if pmid else f"no_id_{record_idx}"

        abstract = str(r.get("abstract", ""))
        if len(abstract) <= cfg.min_abstract_len:
            continue

        chunks = chunk_text(abstract, chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap)
        for chunk_idx, ch in enumerate(chunks):
            ids.append(f"{base_id}_c{chunk_idx}")
            documents.append(ch.text)
            metadatas.append(
                {
                    "pmid": pmid,
                    "title": r.get("title"),
                    "source": r.get("source_file"),
                    "chunk_index": chunk_idx,
                    "chunk_start": ch.start,
                    "chunk_end": ch.end,
                }
            )

            if len(documents) >= cfg.batch_size:
                flush()

    flush()

    print("✅ Chroma indexing complete.")
    print(f"Vector store: {cfg.chroma_dir}")
    print(f"Collection: {cfg.collection_name}")


def main() -> int:
    build_or_update_index(rebuild=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


