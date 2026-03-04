"""
train.py
========
Algorithm 2: PhishGuard-URL Training and Weight Optimisation (offline, run once).
PhishGuard-URL (Molefi, 2026)

Trains RF, GB, and LR classifiers on data/train.csv, optimises ensemble
fusion weights on data/val.csv, and saves all artefacts to models/.

Usage
-----
  python src/train.py
  python src/train.py --train data/train.csv --val data/val.csv --out models/
"""

import argparse
import json
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model  import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics       import log_loss, accuracy_score

RANDOM_SEED = 42

# Exact hyperparameters as reported in Section 7.1 of the paper
RF_PARAMS = dict(
    n_estimators     = 200,
    min_samples_split= 5,
    max_features     = "sqrt",   # floor(sqrt(77)) = 8
    random_state     = RANDOM_SEED,
    n_jobs           = -1,
)

GB_PARAMS = dict(
    n_estimators  = 200,
    learning_rate = 0.1,
    max_depth     = 5,
    random_state  = RANDOM_SEED,
)

LR_PARAMS = dict(
    max_iter     = 1000,
    solver       = "lbfgs",
    random_state = RANDOM_SEED,
    C            = 1.0,
)


def load_split(path: str):
    """Load a CSV split and return (X, y) numpy arrays."""
    df    = pd.read_csv(path)
    y     = df["label"].values
    X     = df.drop(columns=["label"]).select_dtypes(include=[np.number]).values
    return X, y


def optimise_weights(p_rf, p_gb, p_lr, y_val):
    """
    Grid search over the unit simplex Delta^2 for fusion weights.
    Minimises cross-entropy loss on validation set.

    Returns (w_rf, w_gb, w_lr) with sum = 1.
    """
    best_loss    = np.inf
    best_weights = (0.45, 0.40, 0.15)   # initialise at paper values

    step = 0.05
    grid = np.arange(0.0, 1.0 + step, step)

    for w_rf in grid:
        for w_gb in grid:
            w_lr = round(1.0 - w_rf - w_gb, 10)
            if w_lr < 0 or w_lr > 1.0:
                continue
            # Weighted average of probabilities
            p_ens = w_rf * p_rf + w_gb * p_gb + w_lr * p_lr
            # Clip to avoid log(0)
            p_ens = np.clip(p_ens, 1e-15, 1 - 1e-15)
            loss  = log_loss(y_val, p_ens)
            if loss < best_loss:
                best_loss    = loss
                best_weights = (round(w_rf, 4), round(w_gb, 4), round(w_lr, 4))

    return best_weights, best_loss


def train(train_path: str, val_path: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Load data
    # ------------------------------------------------------------------
    print("Loading training data ...")
    X_train, y_train = load_split(train_path)
    X_val,   y_val   = load_split(val_path)

    print(f"  Train: {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
    print(f"  Val  : {X_val.shape[0]:,} samples")

    # ------------------------------------------------------------------
    # Step 2: Fit StandardScaler on TRAINING SET ONLY
    # (same fitted scaler must be used at inference — Algorithm 3, Step 3)
    # ------------------------------------------------------------------
    print("\nFitting StandardScaler on training set ...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)      # transform only, no refit

    scaler_path = os.path.join(out_dir, "scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"  Saved: {scaler_path}")

    # ------------------------------------------------------------------
    # Step 3: Train classifiers
    # ------------------------------------------------------------------
    print("\nTraining Random Forest ...")
    t0 = time.time()
    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(X_train, y_train)
    print(f"  Done in {time.time()-t0:.1f}s  |  "
          f"Train accuracy: {accuracy_score(y_train, rf.predict(X_train)):.4f}")
    joblib.dump(rf, os.path.join(out_dir, "rf_model.joblib"))

    print("Training Gradient Boosting ...")
    t0 = time.time()
    gb = GradientBoostingClassifier(**GB_PARAMS)
    gb.fit(X_train, y_train)
    print(f"  Done in {time.time()-t0:.1f}s  |  "
          f"Train accuracy: {accuracy_score(y_train, gb.predict(X_train)):.4f}")
    joblib.dump(gb, os.path.join(out_dir, "gb_model.joblib"))

    print("Training Logistic Regression (on scaled features) ...")
    t0 = time.time()
    lr = LogisticRegression(**LR_PARAMS)
    lr.fit(X_train_scaled, y_train)
    print(f"  Done in {time.time()-t0:.1f}s  |  "
          f"Train accuracy: {accuracy_score(y_train, lr.predict(X_train_scaled)):.4f}")
    joblib.dump(lr, os.path.join(out_dir, "lr_model.joblib"))

    # ------------------------------------------------------------------
    # Step 4: Obtain validation probability estimates
    # ------------------------------------------------------------------
    print("\nComputing validation predictions ...")
    p_rf_val = rf.predict_proba(X_val)[:, 1]
    p_gb_val = gb.predict_proba(X_val)[:, 1]
    p_lr_val = lr.predict_proba(X_val_scaled)[:, 1]   # scaled input

    print(f"  Val accuracy RF : {accuracy_score(y_val, rf.predict(X_val)):.4f}")
    print(f"  Val accuracy GB : {accuracy_score(y_val, gb.predict(X_val)):.4f}")
    print(f"  Val accuracy LR : {accuracy_score(y_val, lr.predict(X_val_scaled)):.4f}")

    # ------------------------------------------------------------------
    # Step 5: Optimise fusion weights on validation set
    # (Equation 7 in the paper)
    # ------------------------------------------------------------------
    print("\nOptimising fusion weights (grid search on unit simplex) ...")
    (w_rf, w_gb, w_lr), best_loss = optimise_weights(
        p_rf_val, p_gb_val, p_lr_val, y_val
    )

    print(f"  Optimal weights: w_RF={w_rf}, w_GB={w_gb}, w_LR={w_lr}")
    print(f"  Validation cross-entropy: {best_loss:.6f}")

    # Ensemble validation accuracy
    p_ens_val = w_rf * p_rf_val + w_gb * p_gb_val + w_lr * p_lr_val
    y_ens_val = (p_ens_val >= 0.5).astype(int)
    print(f"  Ensemble val accuracy: {accuracy_score(y_val, y_ens_val):.4f}")

    # ------------------------------------------------------------------
    # Step 6: Save fusion weights
    # ------------------------------------------------------------------
    weights = {
        "w_RF": w_rf,
        "w_GB": w_gb,
        "w_LR": w_lr,
        "threshold": 0.5,
        "val_log_loss": round(best_loss, 6),
    }
    weights_path = os.path.join(out_dir, "fusion_weights.json")
    with open(weights_path, "w") as f:
        json.dump(weights, f, indent=2)
    print(f"\nSaved fusion weights: {weights_path}")

    print("\n=== Training complete. Saved artefacts: ===")
    for fname in ["rf_model.joblib", "gb_model.joblib",
                  "lr_model.joblib", "scaler.joblib", "fusion_weights.json"]:
        full = os.path.join(out_dir, fname)
        size = os.path.getsize(full) / 1024
        print(f"  {full}  ({size:.0f} KB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train PhishGuard-URL ensemble (Algorithm 2)."
    )
    parser.add_argument("--train", default="data/train.csv")
    parser.add_argument("--val",   default="data/val.csv")
    parser.add_argument("--out",   default="models/")
    args = parser.parse_args()

    for p in [args.train, args.val]:
        if not os.path.exists(p):
            print(f"ERROR: File not found: {p}")
            print("Run dataset_split.py first.")
            sys.exit(1)

    train(args.train, args.val, args.out)
