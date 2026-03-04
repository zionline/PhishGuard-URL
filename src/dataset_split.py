"""
dataset_split.py
================
Reproducible 70/15/15 stratified dataset split (random seed 42).
PhishGuard-URL (Molefi, 2026)

Produces:
  data/train.csv   -- 10,375 records (70%)
  data/val.csv     -- 2,224 records  (15%)
  data/test.csv    -- 2,224 records  (15%)

Usage
-----
  python src/dataset_split.py --input data/url_data.csv
"""

import argparse
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42


def create_split(input_path: str, output_dir: str) -> None:
    """
    Load dataset, apply stratified 70/15/15 split, save to CSV.

    Parameters
    ----------
    input_path : str
        Path to full dataset CSV. Must contain a 'label' column
        where 0 = legitimate, 1 = phishing.
    output_dir : str
        Directory to write train.csv, val.csv, test.csv.
    """
    print(f"Loading dataset from: {input_path}")
    df = pd.read_csv(input_path)

    if "label" not in df.columns:
        # Try common alternative column names
        for candidate in ["Label", "CLASS_LABEL", "class", "target", "phishing"]:
            if candidate in df.columns:
                df = df.rename(columns={candidate: "label"})
                print(f"  Renamed column '{candidate}' to 'label'")
                break
        else:
            raise ValueError(
                "Could not find a label column. "
                "Expected 'label', 'Label', or 'CLASS_LABEL'."
            )

    n_total     = len(df)
    n_legit     = (df["label"] == 0).sum()
    n_phishing  = (df["label"] == 1).sum()

    print(f"\nDataset summary:")
    print(f"  Total records  : {n_total:,}")
    print(f"  Legitimate (0) : {n_legit:,} ({100*n_legit/n_total:.1f}%)")
    print(f"  Phishing   (1) : {n_phishing:,} ({100*n_phishing/n_total:.1f}%)")

    # --- Step 1: split off 70% train ---
    df_train, df_temp = train_test_split(
        df,
        test_size=0.30,
        stratify=df["label"],
        random_state=RANDOM_SEED,
    )

    # --- Step 2: split remaining 30% equally into val and test (15/15) ---
    df_val, df_test = train_test_split(
        df_temp,
        test_size=0.50,
        stratify=df_temp["label"],
        random_state=RANDOM_SEED,
    )

    # --- Verify counts ---
    print(f"\nSplit results (seed={RANDOM_SEED}):")
    for name, split_df in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
        n  = len(split_df)
        n0 = (split_df["label"] == 0).sum()
        n1 = (split_df["label"] == 1).sum()
        print(f"  {name:<6}: {n:>6,} total  |  {n0:>5,} legitimate  |  {n1:>5,} phishing")

    total_check = len(df_train) + len(df_val) + len(df_test)
    assert total_check == n_total, \
        f"Split total {total_check} != original {n_total}"

    # --- Save ---
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.csv")
    val_path   = os.path.join(output_dir, "val.csv")
    test_path  = os.path.join(output_dir, "test.csv")

    df_train.to_csv(train_path, index=False)
    df_val.to_csv(val_path,     index=False)
    df_test.to_csv(test_path,   index=False)

    print(f"\nSaved:")
    print(f"  {train_path}")
    print(f"  {val_path}")
    print(f"  {test_path}")
    print("\nDone. Use train.csv, val.csv, test.csv in subsequent scripts.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create reproducible 70/15/15 stratified split."
    )
    parser.add_argument(
        "--input",  default="data/url_data.csv",
        help="Path to full dataset CSV (default: data/url_data.csv)"
    )
    parser.add_argument(
        "--output_dir", default="data",
        help="Directory to save split CSVs (default: data/)"
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        print("Please download the UNB dataset first (see data/README_data.md).")
        sys.exit(1)

    create_split(args.input, args.output_dir)
