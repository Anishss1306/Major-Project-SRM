import os
import pandas as pd
import glob

# Configuration
RAW_PATH = os.path.join("data", "raw", "fda")
PROCESSED_PATH = os.path.join("data", "processed")
OUTPUT_FILE = os.path.join(PROCESSED_PATH, "fda_drugs.parquet")


def load_fda_file(filename):
    """
    Helper to find and load a file case-insensitively.
    """
    # Look for the file (e.g., 'Products.txt' or 'products.txt')
    pattern = os.path.join(RAW_PATH, "*")
    files = glob.glob(pattern)

    target_file = None
    for f in files:
        if filename.lower() in os.path.basename(f).lower():
            target_file = f
            break

    if not target_file:
        print(f"⚠️ Warning: Could not find {filename} in {RAW_PATH}")
        return None

    print(f"📖 Loading {os.path.basename(target_file)}...")
    # Drugs@FDA uses tab separation and 'cp1252' encoding (standard for Windows text files)
    try:
        return pd.read_csv(target_file, sep="\t", encoding="cp1252", on_bad_lines='skip')
    except:
        return pd.read_csv(target_file, sep="\t", encoding="utf-8", on_bad_lines='skip')


def main():
    # 1. Load the core tables
    products = load_fda_file("Products.txt")
    applications = load_fda_file("Applications.txt")

    if products is None or applications is None:
        print("❌ Critical files missing. Ensure 'Products.txt' and 'Applications.txt' are in data/raw/fda/")
        return

    # 2. Clean and Rename Columns for consistency
    products = products.rename(columns={
        "DrugName": "drug_name",
        "ActiveIngredient": "active_ingredient",
        "ApplNo": "appl_no",
        "Form": "form",
        "Strength": "strength"
    })

    applications = applications.rename(columns={
        "ApplNo": "appl_no",
        "SponsorName": "sponsor_name"
    })

    # 3. Merge to get Manufacturer info
    print("🔄 Merging Product and Application data...")
    merged = pd.merge(products, applications[['appl_no', 'sponsor_name']], on="appl_no", how="left")

    # 4. Fill missing values
    merged = merged.fillna("Unknown")

    # --- CRITICAL FIX START ---
    # Force all columns to string type.
    # This prevents the "tried to convert to double" error in PyArrow.
    print("🛠️  Normalizing data types...")
    for col in merged.columns:
        merged[col] = merged[col].astype(str)
    # --- CRITICAL FIX END ---

    # 5. Normalize text (Upper case -> Title case for readability)
    merged['drug_name'] = merged['drug_name'].str.title()
    merged['active_ingredient'] = merged['active_ingredient'].str.title()

    # 6. Save
    print(f"💾 Saving {len(merged)} drug records to {OUTPUT_FILE}...")
    merged.to_parquet(OUTPUT_FILE, index=False)
    print("✅ FDA Data Parsing Complete.")


if __name__ == "__main__":
    main()