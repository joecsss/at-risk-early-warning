"""
data_prep.py — โหลดและเตรียมข้อมูลสำหรับระบบเตือนภัยล่วงหน้านักศึกษาเสี่ยง
ค่าเริ่มต้น: ชุดข้อมูล UCI "Predict Students' Dropout and Academic Success" (id=697)

รัน:
    python src/data_prep.py                      # ใช้ฟีเจอร์ทั้งหมด (ภาค 1 + ภาค 2)
    python src/data_prep.py --feature-set early  # ใช้เฉพาะข้อมูลที่รู้ตั้งแต่ต้น (ภาค 1)

ผลลัพธ์: data/processed_train.csv, data/processed_test.csv, data/metadata.json
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from common import (
    DATA_DIR, ID_COL, METADATA_JSON, TARGET_COL, TEST_CSV, TRAIN_CSV,
    save_json, setup_console,
)

RAW_CACHE = os.path.join(DATA_DIR, "raw_uci697.csv")

# คอลัมน์ในไฟล์ต้นฉบับมีชื่อไม่สม่ำเสมอ (สะกดผิด/มีช่องว่างเกิน) — ปรับให้เป็นมาตรฐาน
RENAME_MAP = {
    "Nacionality": "Nationality",
    "Daytime/evening attendance\t": "Daytime/evening attendance",
    "Marital status": "Marital Status",
}


def load_raw(use_cache: bool = True) -> pd.DataFrame:
    """ดึงข้อมูลจาก UCI ผ่าน ucimlrepo แล้วแคชไว้ เพื่อไม่ต้องต่อเน็ตซ้ำทุกครั้ง."""
    if use_cache and os.path.exists(RAW_CACHE):
        print(f"อ่านข้อมูลดิบจากแคช: {RAW_CACHE}")
        return pd.read_csv(RAW_CACHE)

    print("ดาวน์โหลดข้อมูลจาก UCI (id=697) ...")
    from ucimlrepo import fetch_ucirepo

    ds = fetch_ucirepo(id=697)
    X = ds.data.features.copy()
    y = ds.data.targets.copy()
    df = pd.concat([X, y], axis=1)
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(RAW_CACHE, index=False)
    print(f"บันทึกแคชข้อมูลดิบ: {RAW_CACHE}")
    return df


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df.rename(columns={k.strip(): v for k, v in RENAME_MAP.items()})


def make_binary_target(df: pd.DataFrame, target_col: str = "Target") -> pd.DataFrame:
    """นิยาม 'เสี่ยง' = Dropout (1), ไม่เสี่ยง = Enrolled/Graduate (0)."""
    df = df.copy()
    df[TARGET_COL] = (df[target_col].astype(str).str.strip() == "Dropout").astype(int)
    return df.drop(columns=[target_col])


def _safe_ratio(numer: pd.Series, denom: pd.Series) -> pd.Series:
    """หารแบบกันหารศูนย์ (ถ้าตัวหาร = 0 ให้ผลเป็น 0)."""
    return np.where(denom > 0, numer / denom.replace(0, np.nan), 0.0)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """สร้างฟีเจอร์เชิงพฤติกรรม/การเงินที่ตีความได้ง่ายสำหรับอาจารย์ที่ปรึกษา."""
    df = df.copy()

    for sem, tag in ((1, "1st"), (2, "2nd")):
        enrolled = df[f"Curricular units {tag} sem (enrolled)"]
        approved = df[f"Curricular units {tag} sem (approved)"]
        no_eval = df[f"Curricular units {tag} sem (without evaluations)"]
        df[f"pass_rate_{tag}"] = _safe_ratio(approved, enrolled)
        df[f"eval_rate_{tag}"] = _safe_ratio(enrolled - no_eval, enrolled)

    df["grade_change"] = (
        df["Curricular units 2nd sem (grade)"] - df["Curricular units 1st sem (grade)"]
    )
    df["pass_rate_change"] = df["pass_rate_2nd"] - df["pass_rate_1st"]

    # สัญญาณเตือนที่ตีความง่าย
    df["zero_pass_1st"] = (df["Curricular units 1st sem (approved)"] == 0).astype(int)
    df["no_grade_1st"] = (df["Curricular units 1st sem (grade)"] == 0).astype(int)
    df["is_mature"] = (df["Age at enrollment"] > 22).astype(int)

    # ความเสี่ยงด้านการเงิน: มีหนี้ + ค้างค่าเทอม + ไม่มีทุน (0–3)
    df["financial_risk"] = (
        df["Debtor"]
        + (1 - df["Tuition fees up to date"])
        + (1 - df["Scholarship holder"])
    )

    # ผลการเรียนจริงเทียบกับศักยภาพแรกเข้า (สเกลคะแนนสอบเข้า 0–200 -> 0–20)
    df["academic_gap"] = (
        df["Admission grade"] / 10.0 - df["Curricular units 1st sem (grade)"]
    )
    return df


def select_feature_set(df: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    """early = ใช้เฉพาะข้อมูลที่ทราบตั้งแต่ต้น (แรกเข้า + ภาค 1) — เตือนได้เร็วกว่า."""
    if feature_set == "all":
        return df
    drop = [c for c in df.columns if "2nd sem" in c or c.endswith("_2nd")]
    drop += ["grade_change", "pass_rate_change"]
    return df.drop(columns=[c for c in drop if c in df.columns])


def main() -> None:
    setup_console()
    ap = argparse.ArgumentParser(description="เตรียมข้อมูลสำหรับระบบเตือนภัยล่วงหน้า")
    ap.add_argument(
        "--feature-set", choices=["all", "early"], default="all",
        help="all = ใช้ทุกฟีเจอร์ (ภาค 1+2), early = เฉพาะข้อมูลต้นทาง+ภาค 1",
    )
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--refresh", action="store_true", help="ดาวน์โหลดข้อมูลใหม่ ไม่ใช้แคช")
    args = ap.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    df = load_raw(use_cache=not args.refresh)
    df = clean_columns(df)
    df = make_binary_target(df)
    df = engineer_features(df)
    df = select_feature_set(df, args.feature_set)

    # รหัสนักศึกษาจำลอง (ชุดข้อมูล UCI ไม่มี id) — ใช้อ้างอิงในแดชบอร์ดเท่านั้น
    df.insert(0, ID_COL, [f"S{i:04d}" for i in range(1, len(df) + 1)])

    n_missing = int(df.isna().sum().sum())
    if n_missing:
        print(f"เติมค่าที่หายไป {n_missing} ช่อง ด้วยค่ามัธยฐาน")
        num_cols = df.select_dtypes(include="number").columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    train, test = train_test_split(
        df, test_size=args.test_size, stratify=df[TARGET_COL], random_state=args.seed
    )
    train.to_csv(TRAIN_CSV, index=False)
    test.to_csv(TEST_CSV, index=False)

    features = [c for c in df.columns if c not in (ID_COL, TARGET_COL)]
    save_json(METADATA_JSON, {
        "feature_set": args.feature_set,
        "features": features,
        "n_features": len(features),
        "n_train": len(train),
        "n_test": len(test),
        "at_risk_rate": round(float(df[TARGET_COL].mean()), 4),
        "target": TARGET_COL,
        "id_col": ID_COL,
        "source": "UCI id=697 (Predict Students' Dropout and Academic Success)",
        "created_at": pd.Timestamp.now().isoformat(timespec="seconds"),
    })

    print(f"\nfeature_set = {args.feature_set} ({len(features)} ฟีเจอร์)")
    print(f"saved: train={len(train)} test={len(test)} "
          f"at_risk_rate={df[TARGET_COL].mean():.3f}")
    print(f"metadata: {METADATA_JSON}")


if __name__ == "__main__":
    main()
