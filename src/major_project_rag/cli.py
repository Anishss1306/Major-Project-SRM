from __future__ import annotations

import argparse
from dataclasses import asdict

from major_project_rag.ingestion.chroma_index import ChromaIndexConfig, build_or_update_index
from major_project_rag.ingestion.drugbank_vocab import parse_drugbank_vocab
from major_project_rag.ingestion.fda_parse import parse_fda_drugs
from major_project_rag.ingestion.pubmed_extract import extract_pubmed_xml_to_parquet


def _cmd_env_check(_args: argparse.Namespace) -> int:
    import chromadb
    import torch

    print(f"Torch CUDA Available: {torch.cuda.is_available()}")
    print(f"ChromaDB Version: {chromadb.__version__}")
    print("✅ Environment looks ready.")
    return 0


def _cmd_pubmed_extract(_args: argparse.Namespace) -> int:
    extract_pubmed_xml_to_parquet()
    return 0


def _cmd_chroma_index(args: argparse.Namespace) -> int:
    cfg = ChromaIndexConfig(
        batch_size=args.batch_size,
        embedding_model=args.model,
        collection_name=args.collection,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    build_or_update_index(cfg=cfg, rebuild=args.rebuild, limit=args.limit)
    return 0


def _cmd_drugbank_vocab(_args: argparse.Namespace) -> int:
    parse_drugbank_vocab()
    return 0


def _cmd_fda_parse(_args: argparse.Namespace) -> int:
    parse_fda_drugs()
    return 0


def _cmd_print_config(args: argparse.Namespace) -> int:
    cfg = ChromaIndexConfig()
    print(asdict(cfg))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="major_project_rag", description="Major_Project_RAG utilities")
    sub = p.add_subparsers(dest="cmd", required=True)

    env = sub.add_parser("env-check", help="Print environment diagnostics")
    env.set_defaults(func=_cmd_env_check)

    pe = sub.add_parser("pubmed-extract", help="Extract PubMed XML -> data/processed/pubmed_extracted.parquet")
    pe.set_defaults(func=_cmd_pubmed_extract)

    ci = sub.add_parser("chroma-index", help="Embed + upsert PubMed abstracts into ChromaDB")
    ci.add_argument("--rebuild", action="store_true", help="Delete and recreate the collection before indexing")
    ci.add_argument("--limit", type=int, default=None, help="Limit number of records (for testing)")
    ci.add_argument("--batch-size", type=int, default=100, help="Batch size for embedding/upserts")
    ci.add_argument("--chunk-size", type=int, default=800, help="Chunk size (characters) before embedding")
    ci.add_argument("--chunk-overlap", type=int, default=100, help="Chunk overlap (characters)")
    ci.add_argument("--model", default="all-MiniLM-L6-v2", help="SentenceTransformer model name")
    ci.add_argument("--collection", default="pubmed_evidence", help="Chroma collection name")
    ci.set_defaults(func=_cmd_chroma_index)

    dv = sub.add_parser("drugbank-vocab", help="Parse DrugBank vocabulary CSV -> data/processed/drugbank_vocab.parquet")
    dv.set_defaults(func=_cmd_drugbank_vocab)

    fp = sub.add_parser("fda-parse", help="Parse Drugs@FDA tables -> data/processed/fda_drugs.parquet")
    fp.set_defaults(func=_cmd_fda_parse)

    pc = sub.add_parser("print-config", help="Print default indexing config")
    pc.set_defaults(func=_cmd_print_config)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())


