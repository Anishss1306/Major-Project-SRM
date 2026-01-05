import os
import pandas as pd
import glob

# Configuration
RAW_PATH = os.path.join("data", "raw")  # User put it directly in raw
PROCESSED_PATH = os.path.join("data", "processed")
OUTPUT_FILE = os.path.join(PROCESSED_PATH, "drugbank_vocab.parquet")


def main():
    # 1. Find the CSV file
    # DrugBank vocabulary usually comes as 'drugbank vocabulary.csv'
    csv_files = glob.glob(os.path.join(RAW_PATH, "*vocab*.csv"))

    if not csv_files:
        print(f"❌ No vocabulary CSV found in {RAW_PATH}")
        print("   Please ensure 'drugbank vocabulary.csv' is in 'data/raw/'")
        return

    target_file = csv_files[0]
    print(f"📖 Loading {os.path.basename(target_file)}...")

    # 2. Load Data
    # DrugBank CSVs are standard comma-separated
    df = pd.read_csv(target_file)

    # 3. Clean and Select Columns
    # We primarily need: 'Common name', 'Synonyms', 'DrugBank ID'
    required_cols = ['Common name', 'Synonyms', 'DrugBank ID']

    # Check if columns exist (case sensitive)
    available_cols = [c for c in required_cols if c in df.columns]

    if 'Common name' not in available_cols:
        print("⚠️  Error: 'Common name' column not found. Checking file structure...")
        print(f"   Columns found: {df.columns.tolist()}")
        return

    df = df[available_cols].copy()

    # 4. Standardize
    df = df.rename(columns={
        'Common name': 'drug_name',
        'Synonyms': 'synonyms',
        'DrugBank ID': 'drugbank_id'
    })

    # Fill NAs
    df['synonyms'] = df['synonyms'].fillna("")

    # 5. Save
    print(f"💾 Saving {len(df)} vocabulary records to {OUTPUT_FILE}...")
    df.to_parquet(OUTPUT_FILE, index=False)
    print("✅ DrugBank Vocabulary Parsing Complete.")


if __name__ == "__main__":
    main()