"""
common.py — ค่าคงที่และฟังก์ชันช่วยที่ใช้ร่วมกันทุกสคริปต์
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
MODEL_DIR = os.path.join(ROOT, "models")
REPORT_DIR = os.path.join(ROOT, "reports")

TRAIN_CSV = os.path.join(DATA_DIR, "processed_train.csv")
TEST_CSV = os.path.join(DATA_DIR, "processed_test.csv")
METADATA_JSON = os.path.join(DATA_DIR, "metadata.json")
MODEL_FILE = os.path.join(MODEL_DIR, "model.joblib")
METRICS_JSON = os.path.join(MODEL_DIR, "metrics.json")
MODEL_KKU_FILE = os.path.join(MODEL_DIR, "model_kku.joblib")            # ฝึกจากข้อมูล มข. จริง

ID_COL = "student_id"
TARGET_COL = "at_risk"


def setup_console() -> None:
    """บังคับ stdout/stderr เป็น UTF-8 (คอนโซล Windows ค่าเริ่มต้นพิมพ์ภาษาไทยไม่ได้)."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def load_metadata() -> dict:
    if not os.path.exists(METADATA_JSON):
        raise FileNotFoundError(
            f"ไม่พบ {METADATA_JSON} — รัน `python src/data_prep.py` ก่อน"
        )
    with open(METADATA_JSON, encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
