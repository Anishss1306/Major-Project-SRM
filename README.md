# Major_Project_RAG

This repository contains the **data ingestion + indexing** foundation for a Retrieval-Augmented Generation (RAG) workflow focused on **drug interaction evidence**.

## What it does (today)

- **Extract PubMed** article titles/abstracts from XML into a Parquet file.
- **Chunk, embed & index** PubMed abstracts into a persistent **ChromaDB** vector store.
- **Parse DrugBank vocabulary** into a normalization table (canonical name + synonyms).
- **Parse Drugs@FDA** tables into a Parquet file (not yet used by retrieval).
- **Validate query intent** (blocks dosage/diagnosis) and **normalize drug names** from user queries.

> Note: The repo currently does **not** implement the end-user “RAG answer generator” (retrieve + LLM response). It builds the assets you’d use for that.

## New structure (recommended)

All runnable logic is packaged under `src/major_project_rag/`:

- `src/major_project_rag/__main__.py`: enables `python -m major_project_rag ...`
- `src/major_project_rag/cli.py`: CLI entrypoint + subcommands
- `src/major_project_rag/config.py`: shared filesystem paths (raw/processed/chroma)
- `src/major_project_rag/ingestion/`: ingestion + indexing steps
  - `pubmed_extract.py`: PubMed XML -> Parquet
  - `chroma_index.py`: embed + upsert Parquet into Chroma
  - `drugbank_vocab.py`: DrugBank vocabulary CSV -> Parquet
  - `fda_parse.py`: Drugs@FDA tab files -> Parquet
- `src/major_project_rag/rag/`: query-side helpers (no LLM generation yet)
  - `intent_filter.py`: blocks unsafe intents (dosage/diagnosis)
  - `drug_normalization.py`: maps synonyms -> canonical drug names using DrugBank vocab

## Install

```bash
pip install -r requirements.txt
```

## Run (CLI)

From the repo root:

```bash
python -m major_project_rag --help

# 1) Extract PubMed XML -> Parquet
python -m major_project_rag pubmed-extract

# 2) Embed + upsert PubMed abstracts into ChromaDB
python -m major_project_rag chroma-index --rebuild

# (Optional) Tune chunking (characters)
python -m major_project_rag chroma-index --rebuild --chunk-size 800 --chunk-overlap 100

# 3) Parse DrugBank vocabulary CSV -> Parquet
python -m major_project_rag drugbank-vocab

# 4) Parse Drugs@FDA tables -> Parquet
python -m major_project_rag fda-parse

# Checks
python -m major_project_rag env-check
python -m major_project_rag print-config
```

## File-by-file guide (what each file does)

### Core entrypoints

- **`src/major_project_rag/__main__.py`**
  - Lets you run the package as a module: `python -m major_project_rag ...`

- **`src/major_project_rag/cli.py`**
  - Defines the CLI subcommands (extract, index, parse vocab, parse FDA, env-check).
  - This is the recommended way to run all processes.

### Shared config

- **`src/major_project_rag/config.py`**
  - Centralizes filesystem paths used everywhere:
    - raw data under `data/raw/...`
    - processed outputs under `data/processed/...`
    - Chroma persistence under `data/chroma_db/`

### Ingestion + indexing (the “build” side)

- **`src/major_project_rag/ingestion/pubmed_extract.py`**
  - Reads PubMed dumps from `data/raw/pubmed/*.xml`
  - Stream-parses each XML file (memory efficient)
  - Writes `data/processed/pubmed_extracted.parquet` with columns:
    - `pmid`, `title`, `abstract`, `source_file`

- **`src/major_project_rag/ingestion/chroma_index.py`**
  - Loads `data/processed/pubmed_extracted.parquet`
  - Filters short abstracts
  - **Chunks** abstracts (default ~800 chars, 100 overlap)
  - **Embeds** chunks using SentenceTransformers (default `all-MiniLM-L6-v2`)
  - **Upserts** into a persistent Chroma collection (default `pubmed_evidence`) in `data/chroma_db/`
  - This is where your “embedding and upserting” happens.

- **`src/major_project_rag/ingestion/chunking.py`**
  - Implements the chunking phase used by `chroma_index.py`
  - Splits text into overlapping character windows, trying to end on whitespace

- **`src/major_project_rag/ingestion/drugbank_vocab.py`**
  - Finds a DrugBank vocab CSV in `data/raw/` (pattern `*vocab*.csv`)
  - Outputs `data/processed/drugbank_vocab.parquet` with:
    - `drug_name`, `synonyms`, `drugbank_id` (when present)

- **`src/major_project_rag/ingestion/fda_parse.py`**
  - Reads Drugs@FDA tables from `data/raw/fda/` (e.g., `Products.txt`, `Applications.txt`)
  - Normalizes types and outputs `data/processed/fda_drugs.parquet`
  - (Not yet used in retrieval, but prepared for future use.)

### Query-side helpers (the “RAG input” side)

- **`src/major_project_rag/rag/intent_filter.py`**
  - A simple regex-based safety gate:
    - blocks dosage/diagnosis-related queries

- **`src/major_project_rag/rag/drug_normalization.py`**
  - Loads `data/processed/drugbank_vocab.parquet`
  - Builds synonym → canonical mappings
  - Extracts drug terms from user queries and returns canonical names

## Data locations

By default, the CLI expects:

- PubMed XML: `data/raw/pubmed/*.xml`
- DrugBank vocab CSV: `data/raw/*vocab*.csv`
- Drugs@FDA: `data/raw/fda/Products.txt`, `data/raw/fda/Applications.txt`

Outputs:

- PubMed parquet: `data/processed/pubmed_extracted.parquet`
- DrugBank vocab parquet: `data/processed/drugbank_vocab.parquet`
- FDA parquet: `data/processed/fda_drugs.parquet`
- Chroma DB: `data/chroma_db/` (collection `pubmed_evidence`)


