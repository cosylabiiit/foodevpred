import os, warnings, gc
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.model_selection    import StratifiedKFold
from sklearn.preprocessing      import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics            import (accuracy_score, balanced_accuracy_score,
                                        f1_score, matthews_corrcoef,
                                        precision_score, confusion_matrix)
from sklearn.linear_model       import LogisticRegression
from sklearn.svm                import SVC
from sklearn.neural_network     import MLPClassifier
from xgboost                    import XGBClassifier

import tensorflow as tf
from tensorflow.keras.models    import Sequential
from tensorflow.keras.layers    import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.backend   import clear_session

tf.get_logger().setLevel('ERROR')

# ─────────────────────────────────────────────────────────────────────────────
# 1. PATHS
# ─────────────────────────────────────────────────────────────────────────────
FEATURES_PATH = "/kaggle/input/datasets/hsharmaa/food-plm-opt/prott5_RFE.csv"
LABELS_PATH   = "/kaggle/input/datasets/hsharmaa/food-6356/Food_6356.csv"
LABEL_COL_IDX = 1                        
OUTPUT_DIR    = "/kaggle/working"
OUTPUT_FILE   = os.path.join(OUTPUT_DIR, "final_stacked_ensemble_results.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 2. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
X_raw = pd.read_csv(FEATURES_PATH).values.astype(np.float32)
df_labels = pd.read_csv(LABELS_PATH)
y = LabelEncoder().fit_transform(
    df_labels[df_labels.columns[LABEL_COL_IDX]].values)

n_classes = len(np.unique(y))
print(f"Features : {X_raw.shape}")
print(f"Labels   : {len(y)} sequences | {n_classes} classes")
print(f"Counts   : { {int(c): int((y==c).sum()) for c in np.unique(y)} }\n")
assert X_raw.shape[0] == len(y), "Row mismatch between features and labels."

# ─────────────────────────────────────────────────────────────────────────────
# 3. MODELS
# ─────────────────────────────────────────────────────────────────────────────
def get_base_models():
    """
    Returns fresh base model instances.
    Must be called fresh for every inner-fold fit to avoid state leakage.
    """
    return {
        "LR": LogisticRegression(
            C=2.0, max_iter=1000, solver="lbfgs",
            class_weight="balanced", random_state=42, n_jobs=-1),

        "SVM_RBF": SVC(
            C=1.0, kernel="rbf", probability=True,
            class_weight="balanced", random_state=42),

        "MLP": MLPClassifier(
            hidden_layer_sizes=(512, 256, 128), alpha=0.001,
            max_iter=500, early_stopping=True,
            validation_fraction=0.1, random_state=42),
            # Note: MLPClassifier has no class_weight — left unweighted
    }

def get_meta_model():
    """XGBoost meta-classifier — default settings, do not tune further."""
    return XGBClassifier(
        max_depth=3, learning_rate=0.05, n_estimators=100,
        subsample=0.8, eval_metric="mlogloss",
        random_state=42, n_jobs=-1, verbosity=0)

BASE_MODEL_NAMES = list(get_base_models().keys())   

# ─────────────────────────────────────────────────────────────────────────────
# 4. METRICS
# ─────────────────────────────────────────────────────────────────────────────
def get_metrics(y_true, y_pred):
    acc  = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, average="macro")
    pre  = precision_score(y_true, y_pred, average="macro", zero_division=0)
    mcc  = matthews_corrcoef(y_true, y_pred)
    cm   = confusion_matrix(y_true, y_pred)
    specs = []
    for i in range(len(cm)):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - (tp + fp + fn)
        specs.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
    return acc, bacc, f1, float(np.mean(specs)), mcc, pre

def fmt(arr, is_mcc=False):
    m = 1 if is_mcc else 100
    return f"{np.mean(arr) * m:.2f} ± {np.std(arr) * m:.2f}"

# ─────────────────────────────────────────────────────────────────────────────
# 5. CROSS-VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

metric_keys = ["tr_acc","tr_bacc","tr_f1","tr_spec","tr_mcc","tr_pre",
               "ts_acc","ts_bacc","ts_f1","ts_spec","ts_mcc","ts_pre"]
results = {k: [] for k in metric_keys}

print("=" * 60)
print("  Final Stacked Ensemble — 5-Fold CV")
print("=" * 60)

for fold, (tr_idx, ts_idx) in enumerate(outer_cv.split(X_raw, y)):
    print(f"\n── Outer Fold {fold + 1} / 5 ──")

    # ── Split ──────────────────────────────────────────────────────
    X_tr_raw, X_ts_raw = X_raw[tr_idx], X_raw[ts_idx]
    y_tr, y_ts         = y[tr_idx],     y[ts_idx]

    # ── Scale: fit on train fold only ──────────────────────────────
    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr_raw)
    X_ts   = scaler.transform(X_ts_raw)

    # ── Generate OOF meta-features via inner CV ────────────

    meta_tr = np.zeros((len(y_tr), n_classes * len(BASE_MODEL_NAMES)))
    meta_ts = np.zeros((len(y_ts), n_classes * len(BASE_MODEL_NAMES)))

    for mi, model_name in enumerate(BASE_MODEL_NAMES):
        col_s = mi * n_classes
        col_e = col_s + n_classes

        print(f"  Training base model: {model_name} ...", end=" ", flush=True)

        #Inner-CV OOF for meta-TRAIN
        block_tr = np.zeros((len(y_tr), n_classes))
        for inn_tr_idx, inn_val_idx in inner_cv.split(X_tr, y_tr):
            X_in,  X_val = X_tr[inn_tr_idx], X_tr[inn_val_idx]
            y_in         = y_tr[inn_tr_idx]

            m_inner = get_base_models()[model_name]
            m_inner.fit(X_in, y_in)
            block_tr[inn_val_idx] = m_inner.predict_proba(X_val)

        #Full-fold fit for meta-TEST
        m_full = get_base_models()[model_name]
        m_full.fit(X_tr, y_tr)
        block_ts = m_full.predict_proba(X_ts)

        meta_tr[:, col_s:col_e] = block_tr
        meta_ts[:, col_s:col_e] = block_ts
        print("done")

    # ── Fit XGB meta-classifier on OOF meta-features ────────
    print("  Training meta-classifier (XGBoost) ...", end=" ", flush=True)
    meta_clf = get_meta_model()
    meta_clf.fit(meta_tr, y_tr)
    print("done")

    p_tr = meta_clf.predict(meta_tr)
    p_ts = meta_clf.predict(meta_ts)

    tr_m = get_metrics(y_tr, p_tr)
    ts_m = get_metrics(y_ts, p_ts)

    for i, k in enumerate(["acc","bacc","f1","spec","mcc","pre"]):
        results[f"tr_{k}"].append(tr_m[i])
        results[f"ts_{k}"].append(ts_m[i])

    print(f"  Fold {fold+1} → "
          f"Train ACC {tr_m[0]*100:.2f}%  |  "
          f"Test  ACC {ts_m[0]*100:.2f}%  "
          f"F1 {ts_m[2]*100:.2f}%  "
          f"MCC {ts_m[4]:.3f}  "
          f"Gap {(tr_m[0]-ts_m[0])*100:.2f}")

    gc.collect()

# ─────────────────────────────────────────────────────────────────────────────
# 6. FINAL RESULTS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  FINAL 5-FOLD CV RESULTS")
print("=" * 60)

train_acc = np.mean(results["tr_acc"]) * 100
test_acc  = np.mean(results["ts_acc"]) * 100

summary = {
    "Base_Models"   : "LR(C=2) + SVM_RBF(C=1) + MLP(512-256-128)",
    "Meta_Model"    : "XGBoost(depth=3, lr=0.05, n=100, sub=0.8)",
    "Train_ACC"     : fmt(results["tr_acc"]),
    "Train_BACC"    : fmt(results["tr_bacc"]),
    "Train_F1"      : fmt(results["tr_f1"]),
    "Train_SPEC"    : fmt(results["tr_spec"]),
    "Train_MCC"     : fmt(results["tr_mcc"], is_mcc=True),
    "Train_PRE"     : fmt(results["tr_pre"]),
    "Test_ACC"      : fmt(results["ts_acc"]),
    "Test_BACC"     : fmt(results["ts_bacc"]),
    "Test_F1"       : fmt(results["ts_f1"]),
    "Test_SPEC"     : fmt(results["ts_spec"]),
    "Test_MCC"      : fmt(results["ts_mcc"],  is_mcc=True),
    "Test_PRE"      : fmt(results["ts_pre"]),
    "Overfit_Gap"   : round(train_acc - test_acc, 2),
}

for k, v in summary.items():
    print(f"  {k:<20}: {v}")

# Per-fold detail table
fold_detail = pd.DataFrame({
    "Fold"         : list(range(1, 6)),
    "Train_ACC"    : [round(v*100, 2) for v in results["tr_acc"]],
    "Test_ACC"     : [round(v*100, 2) for v in results["ts_acc"]],
    "Test_F1"      : [round(v*100, 2) for v in results["ts_f1"]],
    "Test_MCC"     : [round(v,    4)  for v in results["ts_mcc"]],
    "Overfit_Gap"  : [round((tr-ts)*100, 2)
                      for tr, ts in zip(results["tr_acc"], results["ts_acc"])],
})
print(f"\n  Per-fold breakdown:")
print(fold_detail.to_string(index=False))

# Save
pd.DataFrame([summary]).to_csv(OUTPUT_FILE, index=False)
print(f"\n  Saved → {OUTPUT_FILE}")
