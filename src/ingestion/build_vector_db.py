import os
import chromadb
import pandas as pd
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Configuration
DATA_PATH = os.path.join("data", "processed", "pubmed_extracted.parquet")
DB_PATH = os.path.join("data", "chroma_db")
COLLECTION_NAME = "pubmed_evidence"
BATCH_SIZE = 100  # Number of articles to process at once


def build_vector_db():
    # 1. Check for data
    if not os.path.exists(DATA_PATH):
        print(f"Error: Processed data file not found at {DATA_PATH}")
        return

    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)

    # Filter out empty abstracts
    df = df[df['abstract'].str.len() > 20]
    print(f"Total articles to index: {len(df)}")

    # 2. Initialize Model
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # 3. Initialize ChromaDB
    print(f"Initializing ChromaDB at {DB_PATH}...")
    chroma_client = chromadb.PersistentClient(path=DB_PATH)

    # robustly handle deletion
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
        print("Deleted existing collection to rebuild.")
    except Exception:
        # If it fails (e.g., doesn't exist), we just ignore it and move on
        print("Collection not found, creating a fresh one.")

    collection = chroma_client.create_collection(name=COLLECTION_NAME)

    # 4. Batch Process and Index
    records = df.to_dict('records')

    # OPTIONAL: For testing, uncomment the next line to only index the first 1000 articles
    # records = records[:1000]

    print("Starting Indexing Process...")

    for i in tqdm(range(0, len(records), BATCH_SIZE), desc="Indexing Batches"):
        batch = records[i: i + BATCH_SIZE]

        # Prepare lists for ChromaDB
        ids = [str(r['pmid']) if r['pmid'] else f"no_id_{i + j}" for j, r in enumerate(batch)]
        documents = [r['abstract'] for r in batch]
        metadatas = [{'title': r['title'], 'source': r['source_file']} for r in batch]

        # Generate Embeddings
        embeddings = model.encode(documents).tolist()

        # Add to ChromaDB
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

    print("Vector Database build complete.")
    print(f"Data saved to {DB_PATH}")


if __name__ == "__main__":
    build_vector_db()