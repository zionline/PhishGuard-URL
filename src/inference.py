"""
inference.py
============
Algorithm 3: PhishGuard-URL Real-Time Inference (per URL).
PhishGuard-URL (Molefi, 2026)

Loads pre-trained models and fusion weights saved by train.py,
then classifies a new URL as phishing or legitimate.

IMPORTANT: The StandardScaler is applied using the mean and std
learned from the TRAINING SET only (no refit at inference time).
This is Algorithm 3, Step 3 in the paper.

Usage
-----
  # Single URL from command line:
  python src/inference.py --url "http://example.com/login?id=123"

  # Python API:
  from inference import PhishGuardPredictor
  predictor = PhishGuardPredictor(models_dir="models/")
  result = predictor.predict("http://secure-paypal.verify-account.com/")
  print(result)
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import joblib

# Feature extraction (Algorithm 1)
sys.path.insert(0, os.path.dirname(__file__))
from feature_extraction import extract_features


class PhishGuardPredictor:
    """
    End-to-end PhishGuard-URL predictor.

    Loads all saved artefacts from models_dir and exposes a
    predict() method that takes a raw URL string and returns
    a classification result with probability score.
    """

    def __init__(self, models_dir: str = "models/"):
        self.models_dir = models_dir
        self._load_artefacts()

    def _load_artefacts(self):
        """Load models, scaler, and fusion weights from disk."""
        def _path(fname):
            return os.path.join(self.models_dir, fname)

        missing = [
            f for f in ["rf_model.joblib", "gb_model.joblib",
                         "lr_model.joblib", "scaler.joblib",
                         "fusion_weights.json"]
            if not os.path.exists(_path(f))
        ]
        if missing:
            raise FileNotFoundError(
                f"Missing model files: {missing}\n"
                "Run src/train.py first to generate trained models."
            )

        self.rf     = joblib.load(_path("rf_model.joblib"))
        self.gb     = joblib.load(_path("gb_model.joblib"))
        self.lr     = joblib.load(_path("lr_model.joblib"))
        self.scaler = joblib.load(_path("scaler.joblib"))

        with open(_path("fusion_weights.json")) as f:
            w = json.load(f)
        self.w_rf      = w["w_RF"]
        self.w_gb      = w["w_GB"]
        self.w_lr      = w["w_LR"]
        self.threshold = w.get("threshold", 0.5)

    def predict(self, url: str) -> dict:
        """
        Classify a single URL.

        Parameters
        ----------
        url : str
            Raw URL string to classify.

        Returns
        -------
        dict with keys:
            url          -- original URL
            label        -- "PHISHING" or "LEGITIMATE"
            probability  -- phishing probability in [0, 1]
            p_RF         -- RF phishing probability
            p_GB         -- GB phishing probability
            p_LR         -- LR phishing probability
            latency_ms   -- inference time in milliseconds
        """
        t_start = time.perf_counter()

        # --- Step 1: Extract features (Algorithm 1) ---
        feat_dict = extract_features(url)
        x = np.array(list(feat_dict.values()), dtype=float).reshape(1, -1)

        # --- Step 2: RF and GB predictions (raw features) ---
        p_rf = self.rf.predict_proba(x)[0, 1]
        p_gb = self.gb.predict_proba(x)[0, 1]

        # --- Step 3: Scale features using TRAINING-SET mean/std (no refit) ---
        x_scaled = self.scaler.transform(x)

        # --- Step 4: LR prediction (scaled features) ---
        p_lr = self.lr.predict_proba(x_scaled)[0, 1]

        # --- Step 5: Weighted soft-voting fusion (Equation 6) ---
        p_final = self.w_rf * p_rf + self.w_gb * p_gb + self.w_lr * p_lr

        # --- Step 6: Binary decision ---
        label = "PHISHING" if p_final >= self.threshold else "LEGITIMATE"

        latency_ms = (time.perf_counter() - t_start) * 1000

        return {
            "url":         url,
            "label":       label,
            "probability": round(float(p_final), 6),
            "p_RF":        round(float(p_rf),    6),
            "p_GB":        round(float(p_gb),    6),
            "p_LR":        round(float(p_lr),    6),
            "latency_ms":  round(latency_ms,     3),
        }

    def predict_batch(self, urls) -> list:
        """Classify a list of URLs. Returns list of result dicts."""
        return [self.predict(url) for url in urls]


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PhishGuard-URL: classify a URL (Algorithm 3)."
    )
    parser.add_argument(
        "--url",
        default="http://secure-login-paypal.com/verify?user=abc&token=xyz",
        help="URL to classify"
    )
    parser.add_argument(
        "--models", default="models/",
        help="Directory containing trained model artefacts"
    )
    args = parser.parse_args()

    predictor = PhishGuardPredictor(models_dir=args.models)
    result    = predictor.predict(args.url)

    print("\n" + "=" * 55)
    print("  PhishGuard-URL: Real-Time Inference")
    print("=" * 55)
    print(f"  URL        : {result['url'][:60]}...")
    print(f"  Decision   : {result['label']}")
    print(f"  P(phishing): {result['probability']:.4f}  "
          f"(threshold = 0.50)")
    print("-" * 55)
    print(f"  P_RF = {result['p_RF']:.4f}  "
          f"(w=0.45)  ->  {result['p_RF']*0.45:.4f}")
    print(f"  P_GB = {result['p_GB']:.4f}  "
          f"(w=0.40)  ->  {result['p_GB']*0.40:.4f}")
    print(f"  P_LR = {result['p_LR']:.4f}  "
          f"(w=0.15)  ->  {result['p_LR']*0.15:.4f}")
    print("-" * 55)
    print(f"  Inference time: {result['latency_ms']:.2f} ms")
    print("=" * 55)
