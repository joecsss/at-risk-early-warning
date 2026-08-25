"""
train.py — เทรน/เปรียบเทียบแบบจำลอง + จูนเกณฑ์ตัดสิน (threshold) + บันทึกโมเดลที่ดีที่สุด

หลักการประเมิน (เฟส 1):
  * ข้อมูลไม่สมดุล (เสี่ยง ~32%) จึงเทียบด้วย PR-AUC เป็นหลัก ไม่ใช้ accuracy
  * "เลือกโมเดลที่ recall สูงสุด" ตรง ๆ ไม่ถูกต้อง เพราะโมเดลที่ทายว่าเสี่ยงทุกคนจะได้ recall = 1.0
    -> เลือกโมเดลด้วย PR-AUC จาก cross-validation แล้วค่อยจูน threshold ให้ได้ recall ตามเป้า
  * ประเมินแบบ cross-validation บนชุด train เท่านั้น ชุด test เก็บไว้วัดครั้งเดียวตอนท้าย

รัน:  python src/train.py [--target-recall 0.85] [--cv 5]
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier, HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, fbeta_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import joblib
from common import (
    METRICS_JSON, MODEL_DIR, MODEL_FILE, REPORT_DIR, TARGET_COL,
    TEST_CSV, TRAIN_CSV, load_metadata, save_json, setup_console,
)

# คอลัมน์ที่เป็น "รหัสหมวดหมู่" ไม่ใช่ตัวเลขเชิงปริมาณ — ต้อง one-hot ก่อนเข้าโมเดลเชิงเส้น
NOMINAL_COLS = [
    "Marital Status", "Application mode", "Course", "Previous qualification",
    "Nationality", "Mother's qualification", "Father's qualification",
    "Mother's occupation", "Father's occupation",
]


def load() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list[str]]:
    meta = load_metadata()
    features = meta["features"]
    tr = pd.read_csv(TRAIN_CSV)
    te = pd.read_csv(TEST_CSV)
    missing = [c for c in features if c not in tr.columns]
    if missing:
        raise ValueError(f"ไฟล์ข้อมูลไม่มีคอลัมน์ {missing} — รัน data_prep.py ใหม่")
    return tr[features], tr[TARGET_COL], te[features], te[TARGET_COL], features


def build_preprocessor(features: list[str]) -> ColumnTransformer:
    """one-hot สำหรับรหัสหมวดหมู่ + สเกลสำหรับตัวเลข (ใช้กับโมเดลเชิงเส้น)."""
    cats = [c for c in NOMINAL_COLS if c in features]
    nums = [c for c in features if c not in cats]
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10,
                              sparse_output=False), cats),
        ("num", StandardScaler(), nums),
    ], verbose_feature_names_out=False)


def build_models(features: list[str], seed: int = 42) -> dict[str, Pipeline]:
    models: dict[str, Pipeline] = {
        "logreg": Pipeline([
            ("pre", build_preprocessor(features)),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]),
        "rf": Pipeline([
            ("clf", RandomForestClassifier(
                n_estimators=400, min_samples_leaf=2,
                class_weight="balanced_subsample", random_state=seed, n_jobs=-1)),
        ]),
        # baseline ตามสเกลเลตันเดิม (ไม่จัดการ imbalance) เก็บไว้เทียบให้เห็นผลต่าง
        "gb": Pipeline([("clf", GradientBoostingClassifier(random_state=seed))]),
        # จัดการ imbalance ด้วย class_weight ในตัว (HistGB เร็วกว่าและรองรับ weight)
        "hgb_balanced": Pipeline([
            ("clf", HistGradientBoostingClassifier(
                class_weight="balanced", learning_rate=0.06, max_iter=400,
                early_stopping=True, validation_fraction=0.15, random_state=seed)),
        ]),
    }
    try:  # จัดการ imbalance ด้วย SMOTE (oversampling ภายใน fold เท่านั้น จึงไม่รั่วข้อมูล)
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline

        models["gb_smote"] = ImbPipeline([
            ("smote", SMOTE(random_state=seed)),
            ("clf", GradientBoostingClassifier(random_state=seed)),
        ])
    except ImportError:
        print("ข้าม gb_smote (ไม่พบ imbalanced-learn)")
    return models


def tune_threshold(y_true, proba, target_recall: float) -> tuple[float, float]:
    """คืน (threshold ที่ F2 สูงสุด, threshold ที่ทำให้ recall ถึงเป้าโดย precision ดีที่สุด)."""
    prec, rec, thr = precision_recall_curve(y_true, proba)
    prec, rec = prec[:-1], rec[:-1]  # ให้ยาวเท่ากับ thr
    with np.errstate(divide="ignore", invalid="ignore"):
        f2 = np.nan_to_num(5 * prec * rec / (4 * prec + rec))
    best_f2 = float(thr[int(np.argmax(f2))])

    ok = rec >= target_recall
    best_target = float(thr[int(np.argmax(np.where(ok, prec, -1)))]) if ok.any() else 0.5
    return best_f2, best_target


def evaluate(y_true, proba, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "threshold": round(float(threshold), 4),
        "recall": round(float(recall_score(y_true, pred)), 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred)), 4),
        "f2": round(float(fbeta_score(y_true, pred, beta=2)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4),
        "pr_auc": round(float(average_precision_score(y_true, proba)), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def save_curves(y_true, proba, model_name: str) -> str | None:
    """บันทึกกราฟ ROC / Precision-Recall ของชุดทดสอบไว้ใช้ในรายงาน."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay
    except ImportError:
        return None

    os.makedirs(REPORT_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    RocCurveDisplay.from_predictions(y_true, proba, ax=axes[0], name=model_name)
    axes[0].set_title("ROC curve (test set)")
    PrecisionRecallDisplay.from_predictions(y_true, proba, ax=axes[1], name=model_name)
    axes[1].set_title("Precision-Recall curve (test set)")
    fig.tight_layout()
    path = os.path.join(REPORT_DIR, "model_curves.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main() -> None:
    setup_console()
    ap = argparse.ArgumentParser(description="เทรนและเปรียบเทียบโมเดล")
    ap.add_argument("--target-recall", type=float, default=0.85,
                    help="เป้าหมาย recall ของกลุ่มเสี่ยง (ค่าเริ่มต้น 0.85)")
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--operating-point", choices=["f2", "target_recall"],
                    default="target_recall",
                    help="เลือกจุดทำงานที่จะบันทึกลงโมเดล")
    args = ap.parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)
    Xtr, ytr, Xte, yte, features = load()
    meta = load_metadata()
    print(f"feature_set = {meta['feature_set']} | ฟีเจอร์ {len(features)} ตัว | "
          f"train {len(Xtr)} / test {len(Xte)} | สัดส่วนกลุ่มเสี่ยง {ytr.mean():.3f}\n")

    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.seed)
    results: dict[str, dict] = {}
    models = build_models(features, args.seed)

    print(f"--- ผลประเมินแบบ {args.cv}-fold cross-validation บนชุด train ---")
    header = f"{'model':<14}{'PR-AUC':>9}{'ROC-AUC':>9}{'recall@0.5':>12}{'prec@0.5':>10}"
    print(header)
    print("-" * len(header))
    for name, model in models.items():
        oof = cross_val_predict(model, Xtr, ytr, cv=cv, method="predict_proba")[:, 1]
        thr_f2, thr_target = tune_threshold(ytr, oof, args.target_recall)
        results[name] = {
            "cv": evaluate(ytr, oof, 0.5),
            "threshold_f2": round(thr_f2, 4),
            "threshold_target_recall": round(thr_target, 4),
            "cv_at_f2": evaluate(ytr, oof, thr_f2),
            "cv_at_target": evaluate(ytr, oof, thr_target),
        }
        r = results[name]["cv"]
        print(f"{name:<14}{r['pr_auc']:>9.3f}{r['roc_auc']:>9.3f}"
              f"{r['recall']:>12.3f}{r['precision']:>10.3f}")

    best_name = max(results, key=lambda n: results[n]["cv"]["pr_auc"])
    best_model = models[best_name]
    key = "threshold_f2" if args.operating_point == "f2" else "threshold_target_recall"
    threshold = results[best_name][key]
    print(f"\nโมเดลที่เลือก: {best_name} (PR-AUC สูงสุดจาก CV)")
    print(f"จุดทำงาน: {args.operating_point} -> threshold = {threshold:.3f}")

    best_model.fit(Xtr, ytr)
    proba_te = best_model.predict_proba(Xte)[:, 1]
    test_default = evaluate(yte, proba_te, 0.5)
    test_tuned = evaluate(yte, proba_te, threshold)

    print("\n--- ผลบนชุดทดสอบ (ไม่เคยเห็นมาก่อน) ---")
    for tag, m in (("threshold 0.50", test_default), (f"threshold {threshold:.2f}", test_tuned)):
        cm = m["confusion_matrix"]
        print(f"{tag:<18} recall={m['recall']:.3f} precision={m['precision']:.3f} "
              f"f1={m['f1']:.3f} f2={m['f2']:.3f} pr_auc={m['pr_auc']:.3f}")
        print(f"{'':<18} TP={cm['tp']} FN={cm['fn']} (พลาดกลุ่มเสี่ยง) "
              f"FP={cm['fp']} TN={cm['tn']}")

    curve_path = save_curves(yte, proba_te, best_name)
    if curve_path:
        print(f"\nบันทึกกราฟประเมินผล: {curve_path}")

    joblib.dump({
        "model": best_model,
        "features": features,
        "model_name": best_name,
        "threshold": threshold,
        "operating_point": args.operating_point,
        "feature_set": meta["feature_set"],
        "target_recall": args.target_recall,
        "test_metrics": test_tuned,
    }, MODEL_FILE)

    save_json(METRICS_JSON, {
        "feature_set": meta["feature_set"],
        "n_features": len(features),
        "cv_folds": args.cv,
        "target_recall": args.target_recall,
        "selected_model": best_name,
        "selected_threshold": threshold,
        "cv_results": results,
        "test_at_default_threshold": test_default,
        "test_at_tuned_threshold": test_tuned,
    })
    print(f"บันทึกโมเดล: {MODEL_FILE}")
    print(f"บันทึกผลประเมิน: {METRICS_JSON}")


if __name__ == "__main__":
    main()
