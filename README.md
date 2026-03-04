# PhishGuard-URL: Reproducibility Package

**Paper:** PhishGuard-URL: An Explainable Ensemble Learning Framework for  
URL-Based Phishing Detection in Academic Environments  
**Author:** Molefi, H.B. — Botswana University of Science & Technology  
**Year:** 2026

---

## Repository Structure

```
PhishGuard_release/
├── README.md                   ← This file
├── requirements.txt            ← Python dependencies
├── src/
│   ├── feature_extraction.py   ← Algorithm 1: URL feature extraction
│   ├── train.py                ← Algorithm 2: Training + weight optimisation
│   ├── inference.py            ← Algorithm 3: Real-time per-URL inference
│   ├── evaluate.py             ← Full evaluation (all tables in the paper)
│   └── dataset_split.py        ← Reproducible 70/15/15 stratified split
├── models/
│   └── (saved after running train.py)
│       ├── rf_model.joblib
│       ├── gb_model.joblib
│       ├── lr_model.joblib
│       ├── scaler.joblib
│       └── fusion_weights.json
├── data/
│   └── README_data.md          ← Instructions to download UNB dataset
├── notebooks/
│   └── PhishGuard_demo.ipynb   ← End-to-end walkthrough notebook
└── docs/
    └── feature_description.md  ← Description of all 77 features
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download the dataset
Follow instructions in `data/README_data.md` to download the UNB URL dataset.  
Place the CSV file at: `data/url_data.csv`

### 3. Create the reproducible dataset split
```bash
python src/dataset_split.py
```
This produces `data/train.csv`, `data/val.csv`, `data/test.csv`  
using **stratified random sampling with random seed 42**.

### 4. Train the ensemble
```bash
python src/train.py
```
Saves all trained models and fusion weights to `models/`.

### 5. Evaluate on the test set
```bash
python src/evaluate.py
```
Reproduces all results in Table 3 of the paper.

### 6. Run inference on a new URL
```bash
python src/inference.py --url "http://example-login-secure.com/verify?id=123"
```

---

## Exact Hyperparameters (as reported in the paper)

| Model | Parameter | Value |
|-------|-----------|-------|
| Random Forest | n_estimators | 200 |
| Random Forest | min_samples_split | 5 |
| Random Forest | max_features | sqrt |
| Random Forest | random_state | 42 |
| Gradient Boosting | n_estimators | 200 |
| Gradient Boosting | learning_rate | 0.1 |
| Gradient Boosting | max_depth | 5 |
| Gradient Boosting | random_state | 42 |
| Logistic Regression | max_iter | 1000 |
| Logistic Regression | solver | lbfgs |
| Logistic Regression | random_state | 42 |
| Fusion weights | w_RF | 0.45 |
| Fusion weights | w_GB | 0.40 |
| Fusion weights | w_LR | 0.15 |

**Global random seed:** 42  
**Dataset split:** 70% train / 15% validation / 15% test (stratified)

---

## Dataset

- **Source:** University of New Brunswick (UNB) Canadian Institute for Cybersecurity  
- **URL:** https://www.unb.ca/cic/datasets/  
- **Total records:** 14,823  
- **Features:** 77 (numeric, no missing values)  
- **Classes:** Legitimate (7,464) / Phishing (7,359)

---

## Expected Results (Test Set, n=2,224)

| Metric | Value |
|--------|-------|
| Accuracy | 98.92% |
| Precision | 98.65% |
| Recall | 99.18% |
| F1-Score | 98.92% |
| AUC-ROC | 0.9992 |
| FPR | 1.34% |

---

## Citation

If you use this code, please cite:

```
@article{molefi2026phishguard,
  title     = {PhishGuard-URL: An Explainable Ensemble Learning Framework
               for URL-Based Phishing Detection in Academic Environments},
  author    = {Molefi, H. B.},
  journal   = {IEEE Access},
  year      = {2026}
}
```

---

## License
MIT License. See LICENSE file for details.
