"""
refresh.py — คำสั่งเดียวจบ: แก้โค้ดเสร็จ รันตัวนี้ แล้ว "เล่มรายงาน" อัปเดตตามอัตโนมัติ

ทำอะไรให้บ้าง (ตามลำดับ):
  1) (ถ้าใส่ --prep) เตรียมข้อมูลใหม่           -> data_prep.py
  2) เทรน + เลือกโมเดล + คำนวณผล               -> train.py   (อัปเดต models/*.joblib, metrics.json)
  3) สร้างกราฟ SHAP ใหม่                        -> explain.py (อัปเดต reports/*.png)
  4) อัปเดตตาราง/ผล/รูปสำหรับเล่ม LaTeX         -> make_report.py (อัปเดต reports/latex/*)

ใช้งาน:
  python src/refresh.py               # เทรนใหม่ + อัปเดตทุกอย่าง (ใช้ตอนแก้โมเดล/ข้อมูล)
  python src/refresh.py --no-train    # ไม่เทรนใหม่ ใช้โมเดลเดิม แค่รีเฟรชกราฟ+รายงาน (เร็ว)
  python src/refresh.py --prep        # เตรียมข้อมูลใหม่ด้วย (ตอนเปลี่ยนชุดข้อมูล)

เสร็จแล้วไปคอมไพล์ main.tex ในเทมเพลต เล่มจะตรงกับโค้ดล่าสุด
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def step(title: str, func) -> None:
    print(f"\n{'=' * 60}\n▶ {title}\n{'=' * 60}", flush=True)
    func()


def run_script(name: str) -> None:
    subprocess.run([sys.executable, os.path.join(HERE, name)], check=True)


def make_shap_plots() -> None:
    sys.path.insert(0, HERE)
    from explain import build_default_explainer, save_global_plots
    expl, test, bundle = build_default_explainer()
    X = test[bundle["features"]]
    paths = save_global_plots(expl, X)
    for p in paths:
        print("  saved", os.path.basename(p))


def main() -> None:
    ap = argparse.ArgumentParser(description="อัปเดตโมเดล + เล่มรายงานให้ตรงกันในคำสั่งเดียว")
    ap.add_argument("--no-train", action="store_true", help="ไม่เทรนใหม่ (ใช้โมเดลเดิม)")
    ap.add_argument("--prep", action="store_true", help="เตรียมข้อมูลใหม่ด้วย")
    args = ap.parse_args()

    if args.prep:
        step("เตรียมข้อมูล (data_prep.py)", lambda: run_script("data_prep.py"))
    if not args.no_train:
        step("เทรน + เลือกโมเดล (train.py)", lambda: run_script("train.py"))
    try:
        step("สร้างกราฟ SHAP (explain.py)", make_shap_plots)
    except Exception as e:
        print(f"  ⚠ ข้ามการสร้างกราฟ SHAP ({type(e).__name__}: {e}) — ใช้กราฟเดิมแทน")
    step("อัปเดตชิ้นส่วนรายงาน (make_report.py)", lambda: run_script("make_report.py"))

    print(f"\n{'=' * 60}")
    print("✅ อัปเดตครบ: โมเดล + ตาราง + กราฟ ตรงกับโค้ดล่าสุดแล้ว")
    print("   ต่อไป: คอมไพล์ main.tex ในเทมเพลต -> เล่มจะเป็นเวอร์ชันล่าสุด")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
