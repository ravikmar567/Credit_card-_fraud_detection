"""
Credit Card Fraud Detection — end-to-end training & evaluation pipeline.

Dataset: Kaggle "Credit Card Fraud Detection" dataset — 284,807 real,
anonymized European cardholder transactions from September 2013, with 492
confirmed frauds (0.173% of transactions). Features V1-V28 are PCA components
of the original transaction features (anonymized for confidentiality);
`Amount` and `Time` are the only two untransformed columns.
Source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
(originally published by the ULB Machine Learning Group).

Run:
    python src/train.py
Outputs go to results/ (metrics.json, metrics.md, and PNG plots).
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, classification_report,
)

sys.path.insert(0, os.path.dirname(__file__))
from smote import smote_balance

RANDOM_STATE = 42
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "creditcard.csv")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def load_data():
    log(f"Loading data from {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
    log(f"Loaded {len(df):,} rows, {df.shape[1]} columns.")
    fraud_rate = df["Class"].mean() * 100
    log(f"Fraud cases: {df['Class'].sum()} / {len(df)} ({fraud_rate:.4f}%)")
    return df


def plot_class_distribution(df):
    counts = df["Class"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(["Legit (0)", "Fraud (1)"], counts.values, color=["#2c6e91", "#c0392b"])
    ax.set_yscale("log")
    ax.set_ylabel("Number of transactions (log scale)")
    ax.set_title("Class distribution — 284,807 transactions")
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "class_distribution.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    log(f"Saved {path}")


def plot_confusion(y_true, y_pred, name):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.2, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Legit", "Fraud"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Legit", "Fraud"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {name}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fname = f"confusion_{name.lower().replace(' ', '_')}.png"
    path = os.path.join(RESULTS_DIR, fname)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    log(f"Saved {path}")


def plot_roc(results):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for name, (y_true, y_proba) in results.items():
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — test set")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "roc_curve.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    log(f"Saved {path}")


def evaluate(name, y_true, y_pred, y_proba):
    metrics = {
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1_score": round(f1_score(y_true, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_true, y_proba), 4),
    }
    log(f"{name}: {metrics}")
    plot_confusion(y_true, y_pred, name)
    return metrics


def main():
    df = load_data()
    plot_class_distribution(df)

    X = df.drop(columns=["Class"]).values
    y = df["Class"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    log(f"Train: {X_train.shape[0]:,} rows ({y_train.sum()} fraud) | "
        f"Test: {X_test.shape[0]:,} rows ({y_test.sum()} fraud)")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    log("Balancing training set with SMOTE (minority oversampling to 1:1)...")
    X_train_bal, y_train_bal = smote_balance(X_train_scaled, y_train, random_state=RANDOM_STATE)
    log(f"After SMOTE: {len(y_train_bal):,} rows, "
        f"{int((y_train_bal == 1).sum()):,} fraud / {int((y_train_bal == 0).sum()):,} legit")

    all_metrics = {}
    roc_data = {}

    log("Training Logistic Regression on SMOTE-balanced data...")
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    lr.fit(X_train_bal, y_train_bal)
    y_pred_lr = lr.predict(X_test_scaled)
    y_proba_lr = lr.predict_proba(X_test_scaled)[:, 1]
    all_metrics["Logistic Regression (SMOTE)"] = evaluate("Logistic Regression", y_test, y_pred_lr, y_proba_lr)
    roc_data["Logistic Regression"] = (y_test, y_proba_lr)

    log("Training Random Forest on SMOTE-balanced data...")
    rf = RandomForestClassifier(
        n_estimators=150, max_depth=14, n_jobs=-1, random_state=RANDOM_STATE, class_weight=None
    )
    rf.fit(X_train_bal, y_train_bal)
    y_pred_rf = rf.predict(X_test_scaled)
    y_proba_rf = rf.predict_proba(X_test_scaled)[:, 1]
    all_metrics["Random Forest (SMOTE)"] = evaluate("Random Forest", y_test, y_pred_rf, y_proba_rf)
    roc_data["Random Forest"] = (y_test, y_proba_rf)

    plot_roc(roc_data)

    # Baseline: Random Forest WITHOUT SMOTE, to show why balancing matters
    log("Training baseline Random Forest WITHOUT SMOTE (for comparison)...")
    rf_base = RandomForestClassifier(n_estimators=150, max_depth=14, n_jobs=-1, random_state=RANDOM_STATE)
    rf_base.fit(X_train_scaled, y_train)
    y_pred_base = rf_base.predict(X_test_scaled)
    y_proba_base = rf_base.predict_proba(X_test_scaled)[:, 1]
    all_metrics["Random Forest (no SMOTE, baseline)"] = evaluate(
        "Random Forest (no SMOTE)", y_test, y_pred_base, y_proba_base
    )

    # Feature importance (from the SMOTE-trained RF, our best model)
    feat_names = df.drop(columns=["Class"]).columns
    importances = pd.Series(rf.feature_importances_, index=feat_names).sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    importances.iloc[::-1].plot(kind="barh", ax=ax, color="#2c6e91")
    ax.set_title("Top 10 feature importances — Random Forest")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "feature_importance.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    log(f"Saved {path}")

    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    md_lines = ["# Results\n", "| Model | Precision | Recall | F1-Score | ROC-AUC |",
                "|---|---|---|---|---|"]
    for name, m in all_metrics.items():
        md_lines.append(f"| {name} | {m['precision']} | {m['recall']} | {m['f1_score']} | {m['roc_auc']} |")
    with open(os.path.join(RESULTS_DIR, "metrics.md"), "w") as f:
        f.write("\n".join(md_lines) + "\n")

    log("Done. See results/metrics.json, results/metrics.md, and results/*.png")
    return all_metrics


if __name__ == "__main__":
    main()
