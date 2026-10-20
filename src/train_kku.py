"""
train_kku.py — ฝึกโมเดลจากข้อมูลรูปแบบทะเบียน มข. โดยตรง (ไม่ผ่านชุด UCI)

1 ตัวอย่าง = นักศึกษา 1 คน ณ สิ้นภาค t (ทุกชั้นปี ปี 1 ถึงปี 4+)
คำตอบ (y) = ภาค t+1..t+H ในทะเบียนเป็นรอพินิจหรือพ้นสภาพไหม (ค่าเริ่มต้น H = 2)
ฟีเจอร์ = ข้อมูล มข. ทั้งหมดใน kku_timeline.KKU_FEATURES (รวมที่ UCI ไม่มี เช่น หน่วยกิตช้ากว่าแผน,
          จำนวนภาคที่เคยรอพินิจ, ค่าปรับลงทะเบียนช้า, รอบรับเข้า, รายได้ครอบครัว)

รัน:
    python src/train_kku.py --data data/kku_terms.csv   # -> models/model_kku.joblib
    python src/train_kku.py --db                        # ใช้ข้อมูลทั้งหมดในฐานข้อมูล (data/kku_app.db)

แบ่ง fold ตาม "ตัวนักศึกษา" (GroupKFold) เพื่อไม่ให้ภาคของคนเดียวกันอยู่ทั้งฝั่งฝึกและฝั่งทดสอบ
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import MODEL_DIR, MODEL_KKU_FILE, setup_console
from kku_schema import validate
from kku_timeline import (KKU_CAT, KKU_FEATURE_LABELS, KKU_FEATURES, KKU_NUM,
                          PROBATION_GPAX, make_labels, term_features)

MIN_POSITIVE = 10


def build_model(seed: int = 42) -> Pipeline:
    pre = ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("sc", StandardScaler()),
        ]), KKU_NUM),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="constant", fill_value="ไม่ทราบ")),
            ("oh", OneHotEncoder(handle_unknown="ignore")),
        ]), KKU_CAT),
    ], verbose_feature_names_out=False)
    return Pipeline([("prep", pre),
                     ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                                random_state=seed))])


def threshold_for_recall(y, p, target: float) -> float:
    prec, rec, thr = precision_recall_curve(y, p)
    ok = np.where(rec[:-1] >= target)[0]
    return float(thr[ok[-1]]) if len(ok) else 0.5


def report(y, p, thr, tag: str) -> dict:
    pred = p >= thr
    tp = int((pred & (y == 1)).sum())
    out = {"roc_auc": round(float(roc_auc_score(y, p)), 4),
           "pr_auc": round(float(average_precision_score(y, p)), 4),
           "recall": round(tp / max(int((y == 1).sum()), 1), 4),
           "precision": round(tp / max(int(pred.sum()), 1), 4)}
    print(f"  {tag:<28} ROC-AUC {out['roc_auc']:.3f} · PR-AUC {out['pr_auc']:.3f} · "
          f"recall {out['recall']:.3f} · precision {out['precision']:.3f}")
    return out


def main() -> None:
    setup_console()
    ap = argparse.ArgumentParser(description="ฝึกโมเดลจากข้อมูลทะเบียน มข.")
    ap.add_argument("--data", help="CSV รูปแบบทะเบียน มข. (1 แถว = 1 คน × 1 ภาค)")
    ap.add_argument("--db", action="store_true", help="อ่านข้อมูลจากฐานข้อมูลของระบบแทนไฟล์ CSV")
    ap.add_argument("--horizon", type=int, default=2, help="ทำนายล่วงหน้ากี่ภาค")
    ap.add_argument("--target-recall", type=float, default=0.85)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    if not args.data and not args.db:
        ap.error("ระบุ --data <ไฟล์> หรือ --db")
    if args.db:
        import db
        raw = db.load_students()
    else:
        raw = pd.read_csv(args.data)
    errors = validate(raw)
    if errors:
        raise SystemExit("ข้อมูลไม่ถูกต้อง:\n- " + "\n- ".join(errors[:20]))

    tf = term_features(raw)
    y_all = make_labels(tf, args.horizon)
    mask = y_all.notna()
    X, y = tf.loc[mask, KKU_FEATURES], y_all[mask].astype(int).values
    groups = tf.loc[mask, "student_id"].values
    n_students = len(set(groups))
    print(f"ข้อมูล: {n_students} คน · {len(X)} ตัวอย่าง (คน×ภาค) · เสี่ยง {int(y.sum())} "
          f"({y.mean():.1%}) · ทำนายล่วงหน้า {args.horizon} ภาค")
    if y.sum() < MIN_POSITIVE or (y == 0).sum() < MIN_POSITIVE:
        raise SystemExit(f"ตัวอย่างแต่ละกลุ่มต้องมีอย่างน้อย {MIN_POSITIVE} — ต้องการข้อมูลเพิ่ม "
                         "(หลายรุ่น หรือทั้งคณะ)")

    folds = min(args.folds, n_students)
    oof = cross_val_predict(build_model(), X, y, groups=groups, cv=GroupKFold(folds),
                            method="predict_proba")[:, 1]
    thr = threshold_for_recall(y, oof, args.target_recall)

    print(f"\nผล cross-validation ({folds} fold แบ่งตามนักศึกษา):")
    cv = report(y, oof, thr, f"โมเดล มข. (threshold {thr:.2f})")
    rule = (X["gpax"] < PROBATION_GPAX).astype(float).values
    base = report(y, rule, 0.5, f"เทียบ: แค่ GPAX < {PROBATION_GPAX:.2f}")

    print("\nแยกตามชั้นปี (recall = จับคนที่จะเสี่ยงได้กี่ %):")
    by_year = {}
    for yr in sorted(X["year"].unique()):
        m = (X["year"] == yr).values
        pos = int(y[m].sum())
        if pos:
            rec = float(((oof[m] >= thr) & (y[m] == 1)).sum() / pos)
            by_year[int(yr)] = {"n": int(m.sum()), "positive": pos, "recall": round(rec, 3)}
            print(f"  ปี {int(yr)}: {m.sum():>5} ตัวอย่าง · เสี่ยง {pos:>4} · recall {rec:.3f}")
        else:
            print(f"  ปี {int(yr)}: {m.sum():>5} ตัวอย่าง · ไม่มีกรณีเสี่ยง")

    model = build_model().fit(X, y)
    names = model.named_steps["prep"].get_feature_names_out()
    coef = pd.Series(model.named_steps["clf"].coef_[0], index=names)
    print("\nน้ำหนักสูงสุด 10 อันดับ (บวก = เพิ่มความเสี่ยง):")
    for n, v in coef.reindex(coef.abs().sort_values(ascending=False).index)[:10].items():
        src = n if n in KKU_FEATURES else max(
            (f for f in KKU_CAT if n.startswith(f + "_")), key=len, default=n)
        print(f"  {v:+.3f}  {KKU_FEATURE_LABELS.get(src, src)}"
              + (f" = {n[len(src) + 1:]}" if n != src else ""))

    os.makedirs(MODEL_DIR, exist_ok=True)
    out = MODEL_KKU_FILE
    joblib.dump({
        "model": model, "features": KKU_FEATURES, "threshold": thr,
        "horizon": args.horizon,
        "n_students": n_students, "n_rows": int(len(X)),
        "cv_metrics": cv, "baseline_gpax": base, "by_year": by_year,
        "background": X.sample(min(200, len(X)), random_state=42),
    }, out)
    print(f"\nบันทึกโมเดล: {out}")


if __name__ == "__main__":
    main()
