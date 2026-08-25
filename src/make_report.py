"""
make_report.py — สร้าง "ชิ้นส่วนรายงาน" อัตโนมัติจากผลของโมเดล
รันหลังเทรน/แก้โค้ด (python src/make_report.py) แล้วนำชิ้นส่วนไป \input ในเล่ม LaTeX ได้เลย
เปลี่ยนโค้ด -> รันใหม่ -> ตัวเลข/ตาราง/รูปในรายงานอัปเดตเอง (คุณแค่ตรวจ)

ผลลัพธ์ไปที่ reports/latex/
  - metrics.tex         ตารางผลของโมเดลที่เลือก
  - model_compare.tex   ตารางเปรียบเทียบทุกโมเดล
  - figures/*.png       รูป SHAP (คัดลอกจาก reports/)
"""
from __future__ import annotations
import json
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
METRICS = os.path.join(ROOT, "models", "metrics.json")
REPORTS = os.path.join(ROOT, "reports")
OUT = os.path.join(REPORTS, "latex")

NAME_TH = {"logreg": "Logistic Regression", "logistic": "Logistic Regression",
           "rf": "Random Forest", "gb": "Gradient Boosting",
           "hgb_balanced": "HistGradientBoosting (ถ่วงน้ำหนัก)",
           "gb_smote": "Gradient Boosting + SMOTE"}
METRIC_ROWS = [("recall", "Recall"), ("precision", "Precision"),
               ("f1", "F1-score"), ("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC")]


def fmt(x):
    try:
        return f"{float(x):.2f}"
    except Exception:
        return "-"


def pick_block(model_res: dict) -> dict:
    """เลือกชุดตัวเลขที่ threshold เป้าหมาย ถ้าไม่มีใช้ cv ปกติ."""
    for k in ("cv_at_target", "cv", "test"):
        if isinstance(model_res.get(k), dict):
            return model_res[k]
    return model_res


def latex_table(caption, label, header, rows) -> str:
    spec = "|l|" + "c|" * (len(header) - 1)
    lines = ["\\begin{table}[H]", "\\centering",
             f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             f"\\begin{{tabular}}{{{spec}}}", "\\hline",
             " & ".join(f"\\textbf{{{h}}}" for h in header) + " \\\\ \\hline"]
    for r in rows:
        lines.append(" & ".join(str(c) for c in r) + " \\\\ \\hline")
    lines += ["\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def main():
    with open(METRICS, encoding="utf-8") as f:
        m = json.load(f)
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)

    cv = m.get("cv_results", {})
    selected = m.get("selected_model", next(iter(cv), ""))
    sel_name = NAME_TH.get(selected, selected)
    sel_block = pick_block(cv.get(selected, {}))

    # 1) ตารางผลของโมเดลที่เลือก
    rows = [(th, fmt(sel_block.get(key))) for key, th in METRIC_ROWS]
    with open(os.path.join(OUT, "metrics.tex"), "w", encoding="utf-8") as f:
        f.write(latex_table(
            f"ผลการทดลองของแบบจำลองที่เลือก ({sel_name}, {m.get('cv_folds', 5)}-fold CV)",
            "tab:prelim", ["ตัวชี้วัด", "ค่า"], rows))

    # 2) ตารางเปรียบเทียบทุกโมเดล
    header = ["แบบจำลอง", "Recall", "Precision", "F1", "ROC-AUC", "PR-AUC"]
    crows = [[NAME_TH.get(k, k)] + [fmt(pick_block(res).get(mk)) for mk, _ in METRIC_ROWS]
             for k, res in cv.items()]
    with open(os.path.join(OUT, "model_compare.tex"), "w", encoding="utf-8") as f:
        f.write(latex_table("เปรียบเทียบประสิทธิภาพของแบบจำลอง",
                            "tab:model_compare", header, crows))

    # 3) คัดลอกรูป SHAP
    for png in ("shap_summary.png", "shap_global_bar.png", "model_curves.png"):
        src = os.path.join(REPORTS, png)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(OUT, "figures", png))

    print("เขียนไฟล์แล้ว:")
    print("  -", os.path.join(OUT, "metrics.tex"))
    print("  -", os.path.join(OUT, "model_compare.tex"))
    print("  - รูปใน", os.path.join(OUT, "figures"))
    print(f"โมเดลที่เลือก: {sel_name} | Recall={fmt(sel_block.get('recall'))} "
          f"Precision={fmt(sel_block.get('precision'))}")


if __name__ == "__main__":
    main()
