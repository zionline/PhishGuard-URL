# Dataset Download Instructions

## UNB URL Dataset

The dataset used in this paper is the **University of New Brunswick (UNB)
Canadian Institute for Cybersecurity URL Dataset**.

- **Download URL:** https://www.unb.ca/cic/datasets/
- **Citation:** Mamun et al. (2016), "Detecting Malicious URLs Using Lexical Analysis"
- **Records:** 14,823
- **Features:** 77 (pre-engineered, numeric, no missing values)
- **Label:** 0 = Legitimate, 1 = Phishing

## Steps

1. Go to https://www.unb.ca/cic/datasets/
2. Find the **URL dataset** section
3. Download the CSV file
4. Rename or copy it to this folder as: `data/url_data.csv`

## Expected CSV format

The file should have 78 columns:
- Columns 1–77: numeric URL features
- Column 78: label (0 or 1) — may be named `label`, `Label`, or `CLASS_LABEL`

## Verification

After placing the file, run:
```bash
python -c "import pandas as pd; df=pd.read_csv('data/url_data.csv'); print(df.shape)"
```

Expected output: `(14823, 78)`

## Then run the full pipeline:
```bash
python src/dataset_split.py    # creates train/val/test CSVs
python src/train.py            # trains models, saves to models/
python src/evaluate.py         # reproduces all paper tables
```
