"""
evaluate.py
===========
Full evaluation of PhishGuard-URL on the held-out test set.
PhishGuard-URL (Molefi, 2026)

Reproduces all numerical results in the paper:
  - Table 3: Overall performance (all 8 models)
  - Table 4: Confusion matrix
  - Table 6: 10-fold cross-validation
  - Table 7: Ablation study

Usage
-----
  python src/evaluate.py
  python src/evaluate.py --test data/test.csv --models models/
"""

import argparse
import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble  import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree      import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm       import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    matthews_corrcoef
)

RANDOM_SEED = 42

# Paper fusion weights
W_RF, W_GB, W_LR = 0.45, 0.40, 0.15


def load_split(path):
    df = pd.read_csv(path)
    y  = df["label"].values
    X  = df.drop(columns=["label"]).select_dtypes(include=[np.number]).values
    return X, y


def compute_metrics(y_true, y_pred, y_prob):
    """Compute all 6 metrics used in the paper."""
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    auc  = roc_auc_score(y_true, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr  = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return {
        "Accuracy":  f"{100*acc:.2f}%",
        "Precision": f"{100*prec:.2f}%",
        "Recall":    f"{100*rec:.2f}%",
        "F1-Score":  f"{100*f1:.2f}%",
        "AUC-ROC":   f"{auc:.4f}",
        "FPR":       f"{100*fpr:.2f}%",
    }


def print_table(title, rows, headers):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")
    col_w = [max(len(h), max(len(str(r[i])) for r in rows))
             for i, h in enumerate(headers)]
    fmt   = "  ".join(f"{{:<{w}}}" for w in col_w)
    print(fmt.format(*headers))
    print("-" * 65)
    for row in rows:
        print(fmt.format(*[str(v) for v in row]))
    print("=" * 65)


def evaluate(test_path, train_path, models_dir):

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    X_test, y_test   = load_split(test_path)
    X_train, y_train = load_split(train_path)

    # Fit scaler on train (for LR and SVM)
    scaler = joblib.load(os.path.join(models_dir, "scaler.joblib"))
    X_test_scaled = scaler.transform(X_test)

    # ------------------------------------------------------------------
    # Load pre-trained PhishGuard models
    # ------------------------------------------------------------------
    rf = joblib.load(os.path.join(models_dir, "rf_model.joblib"))
    gb = joblib.load(os.path.join(models_dir, "gb_model.joblib"))
    lr = joblib.load(os.path.join(models_dir, "lr_model.joblib"))

    with open(os.path.join(models_dir, "fusion_weights.json")) as f:
        w = json.load(f)
    w_rf, w_gb, w_lr = w["w_RF"], w["w_GB"], w["w_LR"]

    # ------------------------------------------------------------------
    # Table 3: Overall performance
    # ------------------------------------------------------------------
    print("\nEvaluating all models on test set ...")

    rows = []

    # 1. Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    rows.append(("Naive Bayes", *compute_metrics(
        y_test, nb.predict(X_test), nb.predict_proba(X_test)[:,1]).values()))

    # 2. Logistic Regression
    lr_baseline = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    lr_scaler   = StandardScaler()
    lr_baseline.fit(lr_scaler.fit_transform(X_train), y_train)
    rows.append(("Logistic Regression", *compute_metrics(
        y_test,
        lr_baseline.predict(lr_scaler.transform(X_test)),
        lr_baseline.predict_proba(lr_scaler.transform(X_test))[:,1]
    ).values()))

    # 3. SVM (RBF)
    svm_scaler = StandardScaler()
    svm = SVC(kernel="rbf", C=1.0, probability=True, random_state=RANDOM_SEED)
    svm.fit(svm_scaler.fit_transform(X_train), y_train)
    rows.append(("SVM (RBF)", *compute_metrics(
        y_test,
        svm.predict(svm_scaler.transform(X_test)),
        svm.predict_proba(svm_scaler.transform(X_test))[:,1]
    ).values()))

    # 4. Decision Tree
    dt = DecisionTreeClassifier(max_depth=20, random_state=RANDOM_SEED)
    dt.fit(X_train, y_train)
    rows.append(("Decision Tree", *compute_metrics(
        y_test, dt.predict(X_test), dt.predict_proba(X_test)[:,1]).values()))

    # 5. k-NN (k=5)
    knn_scaler = StandardScaler()
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(knn_scaler.fit_transform(X_train), y_train)
    rows.append(("k-NN (k=5)", *compute_metrics(
        y_test,
        knn.predict(knn_scaler.transform(X_test)),
        knn.predict_proba(knn_scaler.transform(X_test))[:,1]
    ).values()))

    # 6. Random Forest
    rows.append(("Random Forest", *compute_metrics(
        y_test, rf.predict(X_test), rf.predict_proba(X_test)[:,1]).values()))

    # 7. Gradient Boosting
    rows.append(("Gradient Boosting", *compute_metrics(
        y_test, gb.predict(X_test), gb.predict_proba(X_test)[:,1]).values()))

    # 8. PhishGuard-URL Ensemble
    p_rf_test  = rf.predict_proba(X_test)[:, 1]
    p_gb_test  = gb.predict_proba(X_test)[:, 1]
    p_lr_test  = lr.predict_proba(X_test_scaled)[:, 1]
    p_ens_test = w_rf * p_rf_test + w_gb * p_gb_test + w_lr * p_lr_test
    y_ens_test = (p_ens_test >= 0.5).astype(int)

    rows.append(("PhishGuard-URL (Ensemble)", *compute_metrics(
        y_test, y_ens_test, p_ens_test).values()))

    print_table(
        "Table 3: Overall Detection Performance on UNB Test Set (n=2,224)",
        rows,
        ["Model", "Accuracy", "Precision", "Recall", "F1", "AUC-ROC", "FPR"]
    )

    # ------------------------------------------------------------------
    # Table 4: Confusion matrix
    # ------------------------------------------------------------------
    tn, fp, fn, tp = confusion_matrix(y_test, y_ens_test).ravel()
    print(f"\n{'='*45}")
    print("  Table 4: Confusion Matrix — Proposed Ensemble")
    print(f"{'='*45}")
    print(f"  {'':22} Pred Legit   Pred Phish")
    print(f"  {'True Legit':<22} {tn:<12} {fp}")
    print(f"  {'True Phish':<22} {fn:<12} {tp}")
    print(f"{'='*45}")
    print(f"  FP rate : {100*fp/(fp+tn):.2f}%  |  "
          f"FN rate: {100*fn/(fn+tp):.2f}%")

    # ------------------------------------------------------------------
    # Table 6: 10-fold stratified cross-validation (RF)
    # ------------------------------------------------------------------
    print("\nRunning 10-fold stratified cross-validation (RF) ...")
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])

    rf_cv = RandomForestClassifier(
        n_estimators=200, min_samples_split=5,
        max_features="sqrt", random_state=RANDOM_SEED, n_jobs=-1
    )
    cv   = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)
    accs = cross_val_score(rf_cv, X_all, y_all, cv=cv, scoring="accuracy")

    print(f"\n{'='*55}")
    print("  Table 6: 10-Fold Stratified Cross-Validation (RF)")
    print(f"{'='*55}")
    for i, a in enumerate(accs, 1):
        print(f"  Fold {i:>2}: {100*a:.2f}%")
    print(f"  {'Mean':>7}: {100*accs.mean():.2f}%  ±  {100*accs.std():.2f}%")
    print(f"  {'Range':>7}: {100*accs.min():.2f}%  --  {100*accs.max():.2f}%")
    print(f"{'='*55}")

    # ------------------------------------------------------------------
    # Table 7: Ablation study
    # ------------------------------------------------------------------
    print("\nRunning ablation study ...")

    # Feature group indices (matching F_lex, F_rat, F_tok, F_sym, F_ent)
    # These slice positions correspond to the order in feature_extraction.py
    feature_groups = {
        "Entropy (-13)":      list(range(64, 77)),   # F_ent last 13 features
        "Length (-8)":        [0,1,2,3,4,5,6,7],     # F_lex first 8
        "Symbol/Digit (-19)": list(range(45, 64)),    # F_sym 19 features
        "Ratio (-16)":        list(range(15, 31)),    # F_rat 16 features
        "Token (-14)":        list(range(31, 45)),    # F_tok 14 features
    }

    ablation_rows = []
    gb_full = GradientBoostingClassifier(**{
        "n_estimators": 200, "learning_rate": 0.1,
        "max_depth": 5, "random_state": RANDOM_SEED
    })
    gb_full.fit(X_train, y_train)
    baseline_acc = accuracy_score(y_test, gb_full.predict(X_test))
    baseline_f1  = f1_score(y_test, gb_full.predict(X_test))
    ablation_rows.append((
        "Full model (77 features)", 77,
        f"{100*baseline_acc:.2f}%", f"{100*baseline_f1:.2f}%", "---"
    ))

    all_feats = set(range(X_train.shape[1]))
    for group_name, remove_idx in feature_groups.items():
        keep = sorted(all_feats - set(remove_idx))
        gb_abl = GradientBoostingClassifier(**{
            "n_estimators": 200, "learning_rate": 0.1,
            "max_depth": 5, "random_state": RANDOM_SEED
        })
        gb_abl.fit(X_train[:, keep], y_train)
        abl_acc = accuracy_score(y_test, gb_abl.predict(X_test[:, keep]))
        abl_f1  = f1_score(y_test, gb_abl.predict(X_test[:, keep]))
        delta   = 100 * (abl_acc - baseline_acc)
        ablation_rows.append((
            f"w/o {group_name}", len(keep),
            f"{100*abl_acc:.2f}%", f"{100*abl_f1:.2f}%",
            f"{delta:+.2f}%"
        ))

    print_table(
        "Table 7: Ablation Study (GB, feature group removal)",
        ablation_rows,
        ["Configuration", "Feats", "Acc", "F1", "ΔAcc"]
    )

    print("\nEvaluation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate PhishGuard-URL and reproduce paper tables."
    )
    parser.add_argument("--test",   default="data/test.csv")
    parser.add_argument("--train",  default="data/train.csv")
    parser.add_argument("--models", default="models/")
    args = parser.parse_args()

    for p in [args.test, args.train]:
        if not os.path.exists(p):
            print(f"ERROR: File not found: {p}")
            print("Run dataset_split.py and train.py first.")
            sys.exit(1)

    evaluate(args.test, args.train, args.models)
