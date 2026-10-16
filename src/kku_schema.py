"""
kku_schema.py — พจนานุกรมข้อมูล (data dictionary) ตามระบบทะเบียน มข. โดยตรง

ระบบฝั่ง มข. ใช้ชื่อและหน่วยตามทะเบียน (CR, CP, GPA 0–4, ผลการศึกษารายภาค ฯลฯ)
ชุด UCI เป็นแค่ "ตัวแปลง" ไปเข้าโมเดลชั่วคราว (kku_mapping.py) ไม่ได้เป็นตัวกำหนด schema

แต่ละฟิลด์ระบุว่าใช้ทำอะไร:
  U = แปลงเข้าโมเดล UCI ชั่วคราว
  R = ใช้ในกฎเตือนภัย (ใช้ได้ทันที ไม่ต้องฝึก)
  K = ใช้ในโมเดล มข. (train_kku.py — ฝึกเมื่อได้ข้อมูลจริง)

ไฟล์นี้ไม่มีข้อมูลส่วนบุคคล
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from kku_mapping import PARENT_EDU, PARENT_OCC, STATUS_OPTIONS

ADMISSION_OPTIONS = ["TCAS1 แฟ้มสะสมผลงาน", "TCAS2 โควตา", "TCAS3 Admission",
                     "TCAS4 รับตรงอิสระ", "อื่น ๆ", "ไม่ทราบ"]
PROGRAM_TYPE_OPTIONS = ["ภาคปกติ", "โครงการพิเศษ", "ไม่ทราบ"]
PARENTS_MARITAL_OPTIONS = ["อยู่ด้วยกัน", "แยกกันอยู่", "หย่าร้าง", "บิดาหรือมารดาเสียชีวิต",
                           "ไม่ทราบ"]
INCOME_OPTIONS = ["ต่ำกว่า 15,000", "15,000–30,000", "30,001–50,000", "50,001–100,000",
                  "มากกว่า 100,000", "ไม่ทราบ"]       # รายได้ครอบครัวต่อเดือน (บาท)
FUNDING_OPTIONS = ["ผู้ปกครอง", "กยศ./กรอ.", "ทุนการศึกษา", "ทำงานเอง", "ไม่ทราบ"]
DORM_OPTIONS = ["หอพักในมหาวิทยาลัย", "หอพักเอกชน", "บ้านตนเอง/ญาติ", "ไม่ทราบ"]
TERM_RESULT_OPTIONS = ["ปกติ", "รอพินิจ", "พ้นสภาพ", ""]

# name: (คำอธิบาย, หมวด, บังคับ?, ค่าเริ่มต้นเมื่อว่าง, หน้าในทะเบียน, ใช้ทำอะไร)
TERM_FIELDS: dict[str, tuple] = {
    "term_no":          ("ลำดับภาค (1 = ปี1/1, 2 = ปี1/2, 3 = ปี2/1, ...) ไม่นับภาคฤดูร้อน",
                         "ผลการเรียนรายภาค", True, None, "ผลการเรียน", "URK"),
    "term":             ("ภาค/ปีการศึกษา เช่น 1/2566", "ผลการเรียนรายภาค", False, "",
                         "ผลการเรียน", ""),
    "n_courses":        ("จำนวนวิชาที่ลงทะเบียน", "ผลการเรียนรายภาค", True, None,
                         "ผลการเรียน", "UK"),
    "n_f":              ("จำนวนวิชาที่ได้ F", "ผลการเรียนรายภาค", False, 0, "ผลการเรียน", "URK"),
    "n_w":              ("จำนวนวิชาที่ถอน (W)", "ผลการเรียนรายภาค", False, 0, "ผลการเรียน", "URK"),
    "cr":               ("หน่วยกิตที่ลงทะเบียน (CR)", "ผลการเรียนรายภาค", True, None,
                         "ผลการเรียน", "RK"),
    "cp":               ("หน่วยกิตที่ผ่าน (CP)", "ผลการเรียนรายภาค", True, None,
                         "ผลการเรียน", "RK"),
    "gpa":              ("GPA ของภาค (0–4)", "ผลการเรียนรายภาค", True, None, "ผลการเรียน", "URK"),
    "gpax":             ("GPAX สะสม ณ สิ้นภาค (ว่างได้ ระบบคำนวณจาก GPA×CR)",
                         "ผลการเรียนรายภาค", False, math.nan, "ผลการเรียน", "RK"),
    "term_result":      ("ผลการศึกษาของภาคในทะเบียน: ปกติ / รอพินิจ / พ้นสภาพ",
                         "ผลการเรียนรายภาค", False, "", "ผลการเรียน", "RK"),
    "outstanding_debt": ("มีค่าธรรมเนียมค้างชำระในภาคนี้ 1 / 0", "การเงินรายภาค", False, 0,
                         "ภาระค่าใช้จ่าย", "URK"),
    "late_fine":        ("มีค่าปรับลงทะเบียนช้าในภาคนี้ 1 / 0", "การเงินรายภาค", False, 0,
                         "ภาระค่าใช้จ่าย", "URK"),
    "logins":           ("จำนวนครั้งที่เข้าระบบทะเบียนในภาค", "พฤติกรรม", False, math.nan,
                         "สถิติการเข้าใช้", "RK"),
}

PROFILE_FIELDS: dict[str, tuple] = {
    "student_id":        ("รหัสสมมติ (ห้ามใช้ชื่อหรือรหัสจริง)", "ระบุตัว", True, None, "-", ""),
    "status":            ("สถานภาพปัจจุบัน: ปกติ / รอพินิจ / พ้นสภาพ / ไม่ทราบ", "ระบุตัว", False,
                          "ไม่ทราบ", "ระเบียนประวัติ", ""),
    "faculty":           ("คณะ", "การศึกษา", False, "", "ระเบียนประวัติ", ""),
    "program":           ("หลักสูตร/สาขา", "การศึกษา", False, "", "ระเบียนประวัติ", ""),
    "program_type":      ("ภาคปกติ / โครงการพิเศษ", "การศึกษา", False, "ไม่ทราบ",
                          "ระเบียนประวัติ", "K"),
    "admission_method":  ("วิธีรับเข้า (รอบ TCAS)", "การศึกษา", False, "ไม่ทราบ",
                          "ระเบียนประวัติ", "K"),
    "entry_year":        ("ปีการศึกษาที่เข้า (พ.ศ.)", "การศึกษา", False, math.nan,
                          "ระเบียนประวัติ", ""),
    "required_credits":  ("หน่วยกิตตลอดหลักสูตร", "การศึกษา", False, 120, "ตรวจสอบจบ", "RK"),
    "plan_terms":        ("จำนวนภาคตามแผน (ปกติ 8)", "การศึกษา", False, 8, "ตรวจสอบจบ", "RK"),
    "hs_gpax":           ("เกรดเฉลี่ย ม.ปลาย (0–4)", "ภูมิหลัง", True, None, "ระเบียนประวัติ", "UK"),
    "age_at_enrollment": ("อายุตอนเข้าศึกษา", "ภูมิหลัง", False, 18, "ระเบียนประวัติ", "UK"),
    "male":              ("เพศชาย 1 / หญิง 0", "ภูมิหลัง", True, None, "ระเบียนประวัติ", "UK"),
    "single":            ("โสด 1 / อื่น ๆ 0", "ภูมิหลัง", False, 1, "ระเบียนประวัติ", "U"),
    "same_province":     ("ภูมิลำเนาจังหวัดเดียวกับวิทยาเขต 1 / 0", "ภูมิหลัง", False, 0,
                          "ระเบียนประวัติ", "UK"),
    "disability":        ("มีความพิการ 1 / 0", "ภูมิหลัง", False, 0, "ระเบียนประวัติ", "UK"),
    "health_issue":      ("มีโรคประจำตัว/ปัญหาสุขภาพ 1 / 0", "ภูมิหลัง", False, 0,
                          "ระเบียนประวัติ", "K"),
    "dorm":              ("ที่พัก", "ภูมิหลัง", False, "ไม่ทราบ", "ระเบียนประวัติ", "K"),
    "father_education":  ("วุฒิสูงสุดของบิดา", "ครอบครัว", False, "ไม่ทราบ", "ระเบียนประวัติ", "UK"),
    "mother_education":  ("วุฒิสูงสุดของมารดา", "ครอบครัว", False, "ไม่ทราบ", "ระเบียนประวัติ", "UK"),
    "father_occupation": ("อาชีพหลักของบิดา", "ครอบครัว", False, "ไม่ทราบ", "ระเบียนประวัติ", "UK"),
    "mother_occupation": ("อาชีพหลักของมารดา", "ครอบครัว", False, "ไม่ทราบ", "ระเบียนประวัติ", "UK"),
    "parents_marital":   ("สถานภาพสมรสของบิดามารดา", "ครอบครัว", False, "ไม่ทราบ",
                          "ระเบียนประวัติ", "K"),
    "family_income":     ("รายได้ครอบครัวต่อเดือน (ช่วง)", "ครอบครัว", False, "ไม่ทราบ",
                          "ระเบียนประวัติ", "K"),
    "siblings_studying": ("จำนวนพี่น้องที่กำลังศึกษา", "ครอบครัว", False, math.nan,
                          "ระเบียนประวัติ", "K"),
    "funding_source":    ("แหล่งเงินหลักที่ใช้เรียน", "การเงิน", False, "ไม่ทราบ",
                          "ระเบียนประวัติ", "K"),
    "scholarship":       ("ได้รับทุน/กยศ. 1 / 0", "การเงิน", False, 0, "ระเบียนประวัติ", "UK"),
}

CHOICES = {
    "status": STATUS_OPTIONS, "term_result": TERM_RESULT_OPTIONS,
    "program_type": PROGRAM_TYPE_OPTIONS, "admission_method": ADMISSION_OPTIONS,
    "father_education": list(PARENT_EDU), "mother_education": list(PARENT_EDU),
    "father_occupation": list(PARENT_OCC), "mother_occupation": list(PARENT_OCC),
    "parents_marital": PARENTS_MARITAL_OPTIONS, "family_income": INCOME_OPTIONS,
    "funding_source": FUNDING_OPTIONS, "dorm": DORM_OPTIONS,
}
ALL_FIELDS = {**PROFILE_FIELDS, **TERM_FIELDS}
COLUMNS = list(PROFILE_FIELDS) + list(TERM_FIELDS)
NUMERIC_TERM = ["term_no", "n_courses", "n_f", "n_w", "cr", "cp", "gpa", "gpax",
                "outstanding_debt", "late_fine", "logins"]


def dictionary_df() -> pd.DataFrame:
    """ตารางพจนานุกรมข้อมูล ใช้แสดงในแดชบอร์ดและแนบในเล่ม."""
    rows = []
    for name, (desc, group, req, _d, page, use) in ALL_FIELDS.items():
        rows.append({"คอลัมน์": name, "ความหมาย": desc, "หมวด": group,
                     "หน้าในทะเบียน": page, "บังคับ": "✓" if req else "",
                     "โมเดล UCI": "✓" if "U" in use else "",
                     "กฎเตือนภัย": "✓" if "R" in use else "",
                     "โมเดล มข.": "✓" if "K" in use else ""})
    return pd.DataFrame(rows)


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """เติมคอลัมน์ที่ไม่บังคับด้วยค่าเริ่มต้น แปลงชนิดข้อมูล และกระจายโปรไฟล์ให้ทุกแถวของคนเดียวกัน."""
    df = df.copy()
    df["student_id"] = df["student_id"].astype(str).str.strip()
    prof_cols = [c for c in PROFILE_FIELDS if c in df.columns and c != "student_id"]
    if prof_cols:
        first = df.groupby("student_id", sort=False)[prof_cols].transform("first")
        df[prof_cols] = first
    for name, (_desc, _g, _req, default, _p, _u) in ALL_FIELDS.items():
        if name not in df.columns:
            df[name] = default
        elif default is not None and not (isinstance(default, float) and math.isnan(default)):
            blank = df[name].isna() | (df[name].astype(str).str.strip() == "")
            df[name] = df[name].where(~blank, default)
    for c in NUMERIC_TERM + ["hs_gpax", "age_at_enrollment", "male", "single", "same_province",
                             "disability", "health_issue", "scholarship", "required_credits",
                             "plan_terms", "siblings_studying", "entry_year"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["term_result"] = df["term_result"].fillna("").astype(str).str.strip()
    return df[COLUMNS].sort_values(["student_id", "term_no"], kind="stable").reset_index(drop=True)


def validate(df: pd.DataFrame) -> list[str]:
    missing = [c for c, f in ALL_FIELDS.items() if f[2] and c not in df.columns]
    if missing:
        return [f"ขาดคอลัมน์บังคับ: {', '.join(missing)}"]
    d = normalize(df)
    errors = []
    for i, r in d.iterrows():
        who = f"แถวที่ {i + 1} ({r['student_id']})"
        need = [c for c, f in ALL_FIELDS.items() if f[2] and c != "student_id"]
        empty = [c for c in need if pd.isna(r[c])]
        if empty:
            errors.append(f"{who}: ช่องบังคับว่างหรือไม่ใช่ตัวเลข ({', '.join(empty)})")
            continue
        if r["n_f"] + r["n_w"] > r["n_courses"]:
            errors.append(f"{who}: F + W มากกว่าจำนวนวิชาที่ลง")
        if r["cp"] > r["cr"]:
            errors.append(f"{who}: หน่วยกิตที่ผ่าน (CP) มากกว่าที่ลง (CR)")
        for c in ("gpa", "hs_gpax"):
            if not 0 <= r[c] <= 4:
                errors.append(f"{who}: {c} ต้องอยู่ระหว่าง 0–4")
        if r["term_no"] < 1:
            errors.append(f"{who}: term_no ต้องเริ่มที่ 1")
        for c, opts in CHOICES.items():
            if r[c] not in opts:
                errors.append(f"{who}: '{r[c]}' ไม่อยู่ในรายการของ {c}")
    dup = d[d.duplicated(["student_id", "term_no"], keep=False)]
    if len(dup):
        errors.append(f"มีภาคซ้ำ: {', '.join(sorted(set(dup['student_id'])))}")
    return errors


# ------------------------------------------------------------- ข้อมูลตัวอย่าง
def _profile(sid: str, status: str = "ปกติ", **kw) -> dict:
    base = {
        "student_id": sid, "status": status, "faculty": "",
        "program": "", "program_type": "ไม่ทราบ",
        "admission_method": "ไม่ทราบ", "entry_year": math.nan, "required_credits": 120,
        "plan_terms": 8, "hs_gpax": 3.00, "age_at_enrollment": 18, "male": 1, "single": 1,
        "same_province": 1, "disability": 0, "health_issue": 0, "dorm": "ไม่ทราบ",
        "father_education": "ไม่ทราบ", "mother_education": "ไม่ทราบ",
        "father_occupation": "ไม่ทราบ", "mother_occupation": "ไม่ทราบ",
        "parents_marital": "ไม่ทราบ", "family_income": "ไม่ทราบ", "siblings_studying": math.nan,
        "funding_source": "ไม่ทราบ", "scholarship": 0,
    }
    return {**base, **kw}


def _rows(prof: dict, terms: list[tuple]) -> list[dict]:
    """terms: (n_courses, n_f, n_w, cr, cp, gpa, term_result, debt, late_fine)."""
    out = []
    for i, (n, f, w, cr, cp, gpa, res, debt, late) in enumerate(terms):
        out.append({**prof, "term_no": i + 1, "term": "", "n_courses": n, "n_f": f, "n_w": w,
                    "cr": cr, "cp": cp, "gpa": gpa, "gpax": math.nan, "term_result": res,
                    "outstanding_debt": debt, "late_fine": late, "logins": math.nan})
    return out


def template_df() -> pd.DataFrame:
    """ไฟล์ตัวอย่าง 2 ภาค (ค่าเป็นตัวอย่างกลาง ๆ ไม่ใช่ข้อมูลของใคร — แก้ตามทะเบียนจริง)."""
    return pd.DataFrame(_rows(_profile("KKU-001"), [
        (6, 0, 0, 18, 18, 3.00, "ปกติ", 0, 0),
        (7, 0, 0, 21, 21, 3.00, "ปกติ", 0, 0)]), columns=COLUMNS)
