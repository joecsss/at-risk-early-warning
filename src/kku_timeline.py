"""
kku_timeline.py — ประเมินความเสี่ยงทุกภาคการศึกษา จากข้อมูลรูปแบบทะเบียน มข. (kku_schema.py)

แต่ละภาคใช้ 3 ชั้นประกอบกัน แล้วเอาระดับที่รุนแรงที่สุด:
  1. โมเดล — ถ้ามีโมเดล มข. (models/model_kku.joblib จาก train_kku.py) ใช้ตัวนั้น
     ถ้ายังไม่มี ใช้โมเดล UCI ชั่วคราว โดยแปลง "2 ภาคล่าสุด" ผ่าน kku_mapping.py
  2. กฎจากข้อมูล มข. — ผลการศึกษาในทะเบียน, หน่วยกิตสะสมเทียบแผน, F/W, ค้างชำระ,
     ค่าปรับลงทะเบียนช้า, การเข้าระบบทะเบียนลดลง (ใช้ได้ทันที ไม่ต้องฝึก)
  3. แนวโน้มและเกณฑ์ GPAX — เกรดดิ่ง/ลดต่อเนื่อง, GPAX ต่ำกว่าเกณฑ์รอพินิจ/พ้นสภาพ

ไฟล์นี้ไม่มีข้อมูลส่วนบุคคล
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from common import ID_COL
from kku_mapping import AT_RISK_STATUS, KKU_COLUMNS, kku_to_uci
from kku_schema import normalize

# ------------------------------------------------------------------- เกณฑ์
# ต้องตรวจกับระเบียบ มข. ว่าด้วยการศึกษาระดับปริญญาตรี ฉบับที่ใช้กับรุ่นนั้น ๆ
PROBATION_GPAX = 2.00      # ต่ำกว่านี้ = เข้าเกณฑ์รอพินิจ
DISMISS_GPAX = 1.50        # ต่ำกว่านี้ = เสี่ยงพ้นสภาพ
WATCH_GPAX = 2.25          # เข้าใกล้เกณฑ์รอพินิจ
GPA_DROP = 0.50            # เกรดภาคนี้ต่ำกว่าภาคก่อนตั้งแต่ 0.50 = ดิ่ง
FAIL_W_LIMIT = 2           # F + W ในภาคเดียวตั้งแต่ 2 วิชา
CREDIT_BEHIND = 9          # หน่วยกิตสะสมช้ากว่าแผนตั้งแต่ 9 หน่วยกิต (~3 วิชา)
LOGIN_DROP = 0.5           # เข้าระบบทะเบียนลดลงเกินครึ่งจากภาคก่อน
WATCH_FACTOR = 0.75        # ความน่าจะเป็นจากโมเดล >= 75% ของ threshold = เฝ้าระวัง

LEVELS = ["ปกติ", "เฝ้าระวัง", "เสี่ยง", "เสี่ยงสูง"]
LEVEL_COLORS = {"ปกติ": "#7A8B99", "เฝ้าระวัง": "#4C78A8",
                "เสี่ยง": "#E8A33D", "เสี่ยงสูง": "#D64545"}

# ---------------------------------------------------- ฟีเจอร์ของโมเดล มข.
KKU_NUM = ["term_no", "year", "gpa", "gpa_prev", "gpa_change", "gpax", "gpax_change",
           "n_f", "n_w", "course_pass_rate", "credit_pass_rate", "cp_cum", "credits_behind",
           "terms_on_probation", "on_probation_now", "outstanding_debt", "late_fine",
           "logins_change", "hs_gpax", "age_at_enrollment", "male", "same_province",
           "disability", "health_issue", "scholarship", "siblings_studying"]
KKU_CAT = ["program_type", "admission_method", "family_income", "funding_source", "dorm",
           "parents_marital", "father_education", "mother_education"]
KKU_FEATURES = KKU_NUM + KKU_CAT

KKU_FEATURE_LABELS = {
    "term_no": "ลำดับภาค", "year": "ชั้นปี", "gpa": "GPA ภาคนี้", "gpa_prev": "GPA ภาคก่อน",
    "gpa_change": "GPA เปลี่ยนจากภาคก่อน", "gpax": "GPAX สะสม",
    "gpax_change": "GPAX เปลี่ยนจากภาคก่อน", "n_f": "จำนวนวิชาที่ได้ F",
    "n_w": "จำนวนวิชาที่ถอน (W)", "course_pass_rate": "สัดส่วนวิชาที่ผ่าน",
    "credit_pass_rate": "สัดส่วนหน่วยกิตที่ผ่าน (CP/CR)", "cp_cum": "หน่วยกิตสะสมที่ผ่าน",
    "credits_behind": "หน่วยกิตที่ช้ากว่าแผน", "terms_on_probation": "จำนวนภาคที่เคยรอพินิจ",
    "on_probation_now": "กำลังรอพินิจ", "outstanding_debt": "ค้างชำระค่าธรรมเนียม",
    "late_fine": "ค่าปรับลงทะเบียนช้า", "logins_change": "การเข้าระบบทะเบียนเปลี่ยนไป",
    "hs_gpax": "เกรดเฉลี่ย ม.ปลาย", "age_at_enrollment": "อายุตอนเข้า", "male": "เพศชาย",
    "same_province": "ภูมิลำเนาจังหวัดเดียวกับวิทยาเขต", "disability": "ความพิการ",
    "health_issue": "ปัญหาสุขภาพ", "scholarship": "ได้รับทุน/กยศ.",
    "siblings_studying": "พี่น้องที่กำลังเรียน", "program_type": "ภาคปกติ/โครงการพิเศษ",
    "admission_method": "รอบรับเข้า", "family_income": "รายได้ครอบครัว",
    "funding_source": "แหล่งเงินที่ใช้เรียน", "dorm": "ที่พัก",
    "parents_marital": "สถานภาพบิดามารดา", "father_education": "วุฒิบิดา",
    "mother_education": "วุฒิมารดา",
}


KKU_INTERVENTIONS = {
    "gpa": "GPA ภาคนี้ต่ำ — นัดพบวางแผนการเรียน ติวรายวิชาที่อ่อน",
    "gpa_prev": "GPA ภาคก่อนต่ำ — ติดตามต่อเนื่องว่าภาคนี้ฟื้นหรือยัง",
    "gpa_change": "GPA ลดลงจากภาคก่อน — ค้นหาสาเหตุ (สุขภาพ/การเงิน/ครอบครัว/ภาระงาน) โดยเร็ว",
    "gpax": "GPAX ใกล้หรือต่ำกว่าเกณฑ์ — วางแผนยกระดับเกรด เลือกลงวิชาให้เหมาะสม",
    "gpax_change": "GPAX ลดลง — ทบทวนภาระวิชาที่ลงทะเบียนภาคถัดไป",
    "n_f": "มีวิชาที่ได้ F — วางแผนลงซ้ำและติวเสริมก่อนลงใหม่",
    "n_w": "ถอนหลายวิชา — คุยเรื่องการจัดภาระเรียนและเหตุที่ต้องถอน",
    "course_pass_rate": "ผ่านวิชาน้อย — จัดพี่เลี้ยง/ติวเตอร์ และติดตามรายสัปดาห์",
    "credit_pass_rate": "หน่วยกิตที่ผ่านน้อยกว่าที่ลง — ทบทวนแผนการเรียน",
    "credits_behind": "หน่วยกิตช้ากว่าแผน — เสี่ยงจบไม่ทัน ตรวจแผนการเรียนกับหน้าตรวจสอบจบ",
    "terms_on_probation": "เคยรอพินิจหลายภาค — ติดตามใกล้ชิดเป็นพิเศษ",
    "on_probation_now": "กำลังรอพินิจ — ต้องยกระดับ GPAX ภาคถัดไป วางแผนร่วมกับอาจารย์ที่ปรึกษา",
    "outstanding_debt": "ค้างชำระ — ประสานงานกองทุน/ผ่อนผัน/ทุนฉุกเฉิน",
    "late_fine": "ลงทะเบียน/ชำระช้า — สอบถามปัญหาการเงินหรือการจัดการเวลา",
    "logins_change": "เข้าระบบทะเบียนน้อยลงมาก — อาจขาดการติดตามการเรียน ติดต่อสอบถาม",
    "scholarship": "ไม่มีทุน — ตรวจสอบสิทธิ์ทุนและ กยศ.",
    "funding_source": "แหล่งเงินที่ใช้เรียนอาจไม่มั่นคง — แนะนำทุน/งานพาร์ตไทม์ในมหาวิทยาลัย",
    "family_income": "รายได้ครอบครัวจำกัด — แนะนำทุนและ กยศ.",
    "health_issue": "มีปัญหาสุขภาพ — ประสานหน่วยบริการสุขภาพ/ให้คำปรึกษา",
    "dorm": "เรื่องที่พักอาจกระทบการเรียน — สอบถามการเดินทางและสภาพแวดล้อม",
    "same_province": "มาจากต่างจังหวัด — ติดตามการปรับตัวและเครือข่ายเพื่อน",
}

# ค่าจริงที่ "น่าห่วง" ของฟีเจอร์ มข. — ใช้คู่กับ SHAP ก่อนสร้างข้อเสนอแนะ (ดู feature_labels.should_suggest)
KKU_CONCERN = {
    "gpa": lambda r: r["gpa"] < WATCH_GPAX,
    "gpa_prev": lambda r: r["gpa_prev"] < WATCH_GPAX,
    "gpa_change": lambda r: r["gpa_change"] < 0,
    "gpax": lambda r: r["gpax"] < WATCH_GPAX,
    "gpax_change": lambda r: r["gpax_change"] < 0,
    "n_f": lambda r: r["n_f"] >= 1,
    "n_w": lambda r: r["n_w"] >= 1,
    "course_pass_rate": lambda r: r["course_pass_rate"] < 1,
    "credit_pass_rate": lambda r: r["credit_pass_rate"] < 1,
    "credits_behind": lambda r: r["credits_behind"] > 0,
    "terms_on_probation": lambda r: r["terms_on_probation"] >= 1,
    "on_probation_now": lambda r: r["on_probation_now"] == 1,
    "outstanding_debt": lambda r: r["outstanding_debt"] == 1,
    "late_fine": lambda r: r["late_fine"] == 1,
    "logins_change": lambda r: r["logins_change"] < 0,
    "scholarship": lambda r: r["scholarship"] == 0,
    "health_issue": lambda r: r["health_issue"] == 1,
}

# ข้อความข้อเสนอแนะแบบ มข. สำหรับฟีเจอร์ UCI (ตอนใช้โมเดล UCI ชั่วคราวกับข้อมูล มข.)
# เช่น W ของ มข. คือถอนวิชา ไม่ใช่ "ขาดสอบ" แบบชุด UCI และ "ภาค 1/2" คือภาคก่อน/ภาคล่าสุด
UCI_TIPS_IN_KKU = {
    "Curricular units 1st sem (without evaluations)": KKU_INTERVENTIONS["n_w"],
    "Curricular units 2nd sem (without evaluations)": KKU_INTERVENTIONS["n_w"],
    "eval_rate_1st": KKU_INTERVENTIONS["n_w"],
    "eval_rate_2nd": KKU_INTERVENTIONS["n_w"],
    "Curricular units 1st sem (approved)": KKU_INTERVENTIONS["course_pass_rate"],
    "Curricular units 2nd sem (approved)": KKU_INTERVENTIONS["course_pass_rate"],
    "pass_rate_1st": KKU_INTERVENTIONS["course_pass_rate"],
    "pass_rate_2nd": KKU_INTERVENTIONS["course_pass_rate"],
    "Curricular units 1st sem (grade)": KKU_INTERVENTIONS["gpa_prev"],
    "Curricular units 2nd sem (grade)": KKU_INTERVENTIONS["gpa"],
    "grade_change": KKU_INTERVENTIONS["gpa_change"],
    "pass_rate_change": "สัดส่วนวิชาที่ผ่านลดลงจากภาคก่อน — ค้นหาสาเหตุโดยเร็ว",
    "zero_pass_1st": "ไม่ผ่านสักวิชาในภาคก่อน — ควรพบอาจารย์ที่ปรึกษาโดยด่วน",
    "no_grade_1st": "GPA ภาคก่อนเป็น 0 — ตรวจสอบว่าขาดเรียน ลาพัก หรือมีปัญหาการลงทะเบียน",
    "Debtor": KKU_INTERVENTIONS["outstanding_debt"],
    "Tuition fees up to date": KKU_INTERVENTIONS["late_fine"],
    "Scholarship holder": KKU_INTERVENTIONS["scholarship"],
}


def term_label(t: int) -> str:
    return f"ปี{(int(t) - 1) // 2 + 1}/{(int(t) - 1) % 2 + 1}"


# --------------------------------------------------------- ฟีเจอร์รายภาค
def term_features(raw: pd.DataFrame) -> pd.DataFrame:
    """ข้อมูลรูปแบบทะเบียน -> 1 แถวต่อคนต่อภาค พร้อมฟีเจอร์แนวโน้ม/ความก้าวหน้า."""
    d = normalize(raw)
    out = []
    for sid, g in d.groupby("student_id", sort=False):
        g = g.sort_values("term_no").reset_index(drop=True).copy()
        # หน่วยกิตที่นำมาคิดเกรด (CA) ≈ CR — ใบเกรด มข. ไม่นับวิชา W ใน CR อยู่แล้ว
        # (เดิมลบ 3×W ซ้ำ ทำให้ GPAX สำรองต่ำกว่าจริง 0.02) · คลาดได้เล็กน้อยเฉพาะภาคที่มีวิชา AU/S
        # ค่าสำรองนี้ใช้เฉพาะเมื่อไม่ได้กรอก GPAX — ควรกรอก GPAX จากใบเกรดเสมอ
        w = g["cr"].clip(lower=1)
        calc = (g["gpa"] * w).cumsum() / w.cumsum()
        g["gpax"] = g["gpax"].fillna(calc).round(2)
        g["gpa_prev"] = g["gpa"].shift(1)
        g["gpa_change"] = (g["gpa"] - g["gpa_prev"]).fillna(0.0)
        g["gpax_change"] = g["gpax"].diff().fillna(0.0)
        g["year"] = (g["term_no"] - 1) // 2 + 1
        g["course_pass_rate"] = np.where(g["n_courses"] > 0,
                                         (g["n_courses"] - g["n_f"] - g["n_w"]) / g["n_courses"], 0)
        g["credit_pass_rate"] = np.where(g["cr"] > 0, g["cp"] / g["cr"], 0)
        g["cp_cum"] = g["cp"].cumsum()
        expected = g["required_credits"] / g["plan_terms"] * g["term_no"]
        g["credits_behind"] = (expected - g["cp_cum"]).clip(lower=0).round(0)
        g["on_probation_now"] = (g["term_result"] == "รอพินิจ").astype(int)
        g["terms_on_probation"] = g["on_probation_now"].cumsum()
        prev_login = g["logins"].shift(1)
        g["logins_change"] = np.where(prev_login > 0, g["logins"] / prev_login - 1, np.nan)
        out.append(g)
    return pd.concat(out, ignore_index=True) if out else d


# ------------------------------------------------------- ตัวแปลงเข้าโมเดล UCI
def uci_windows(tf: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    """ภาค (t-1, t) -> ฟีเจอร์ UCI 1 แถว (เริ่มที่ภาค 2) มีคอลัมน์ term_no ไว้จับคู่."""
    rows, terms = [], []
    for _sid, g in tf.groupby("student_id", sort=False):
        g = g.reset_index(drop=True)
        for i in range(1, len(g)):
            a, b = g.iloc[i - 1], g.iloc[i]
            row = {c: b[c] for c in KKU_COLUMNS if c in b.index}
            row["status"] = b["status"] if b["status"] in ("ปกติ", "รอพินิจ", "พ้นสภาพ",
                                                            "ไม่ทราบ") else "ไม่ทราบ"
            row["has_outstanding_debt"] = int(b["outstanding_debt"])
            row["tuition_paid_on_time"] = int(1 - b["late_fine"])
            for s, src in ((1, a), (2, b)):
                row[f"sem{s}_courses_enrolled"] = int(src["n_courses"])
                row[f"sem{s}_courses_passed"] = int(src["n_courses"] - src["n_f"] - src["n_w"])
                row[f"sem{s}_courses_w"] = int(src["n_w"])
                row[f"sem{s}_gpa"] = float(src["gpa"])
            rows.append(row)
            terms.append(int(b["term_no"]))
    if not rows:
        return pd.DataFrame()
    uci = kku_to_uci(pd.DataFrame(rows, columns=list(KKU_COLUMNS)), train)
    uci["term_no"] = terms
    return uci


# ภาษา มข. สำหรับฟีเจอร์ UCI: ฟีเจอร์ -> (ชื่อแบบ มข., ฟังก์ชันดึงค่าจากภาคก่อน a / ภาคล่าสุด b)
UCI_TO_KKU = {
    "Curricular units 1st sem (grade)": ("GPA ภาคก่อน", lambda a, b: f"{a['gpa']:.2f}"),
    "Curricular units 2nd sem (grade)": ("GPA ภาคล่าสุด", lambda a, b: f"{b['gpa']:.2f}"),
    "Curricular units 1st sem (enrolled)": ("วิชาที่ลง ภาคก่อน", lambda a, b: int(a["n_courses"])),
    "Curricular units 2nd sem (enrolled)": ("วิชาที่ลง ภาคล่าสุด", lambda a, b: int(b["n_courses"])),
    "Curricular units 1st sem (evaluations)": ("วิชาที่ไม่ถอน ภาคก่อน",
                                               lambda a, b: int(a["n_courses"] - a["n_w"])),
    "Curricular units 2nd sem (evaluations)": ("วิชาที่ไม่ถอน ภาคล่าสุด",
                                               lambda a, b: int(b["n_courses"] - b["n_w"])),
    "Curricular units 1st sem (approved)": ("วิชาที่ผ่าน ภาคก่อน",
                                            lambda a, b: int(a["n_courses"] - a["n_f"] - a["n_w"])),
    "Curricular units 2nd sem (approved)": ("วิชาที่ผ่าน ภาคล่าสุด",
                                            lambda a, b: int(b["n_courses"] - b["n_f"] - b["n_w"])),
    "Curricular units 1st sem (without evaluations)": ("วิชาที่ถอน (W) ภาคก่อน",
                                                       lambda a, b: int(a["n_w"])),
    "Curricular units 2nd sem (without evaluations)": ("วิชาที่ถอน (W) ภาคล่าสุด",
                                                       lambda a, b: int(b["n_w"])),
    "pass_rate_1st": ("สัดส่วนวิชาที่ผ่าน ภาคก่อน", lambda a, b: f"{a['course_pass_rate']:.0%}"),
    "pass_rate_2nd": ("สัดส่วนวิชาที่ผ่าน ภาคล่าสุด", lambda a, b: f"{b['course_pass_rate']:.0%}"),
    "eval_rate_1st": ("สัดส่วนวิชาที่ไม่ถอน ภาคก่อน",
                      lambda a, b: f"{1 - a['n_w'] / max(a['n_courses'], 1):.0%}"),
    "eval_rate_2nd": ("สัดส่วนวิชาที่ไม่ถอน ภาคล่าสุด",
                      lambda a, b: f"{1 - b['n_w'] / max(b['n_courses'], 1):.0%}"),
    "grade_change": ("GPA เปลี่ยนจากภาคก่อน", lambda a, b: f"{b['gpa'] - a['gpa']:+.2f}"),
    "pass_rate_change": ("สัดส่วนวิชาที่ผ่านเปลี่ยนไป",
                         lambda a, b: f"{b['course_pass_rate'] - a['course_pass_rate']:+.0%}"),
    "zero_pass_1st": ("ไม่ผ่านสักวิชา ภาคก่อน",
                      lambda a, b: "ใช่" if a["n_courses"] - a["n_f"] - a["n_w"] == 0 else "ไม่ใช่"),
    "no_grade_1st": ("GPA ภาคก่อนเป็น 0", lambda a, b: "ใช่" if a["gpa"] == 0 else "ไม่ใช่"),
    "Previous qualification (grade)": ("เกรดเฉลี่ย ม.ปลาย", lambda a, b: f"{b['hs_gpax']:.2f}"),
    "Debtor": ("ค้างชำระค่าธรรมเนียม ภาคล่าสุด",
               lambda a, b: "ใช่" if b["outstanding_debt"] else "ไม่ใช่"),
    "Tuition fees up to date": ("ไม่มีค่าปรับลงทะเบียนช้า ภาคล่าสุด",
                                lambda a, b: "ใช่" if not b["late_fine"] else "ไม่ใช่"),
    "Scholarship holder": ("ได้รับทุน/กยศ.", lambda a, b: "ใช่" if b["scholarship"] else "ไม่ใช่"),
    "financial_risk": ("ภาระการเงิน (ค้างชำระ+ค่าปรับ+ไม่มีทุน)",
                       lambda a, b: int(b["outstanding_debt"] + b["late_fine"]
                                        + (1 - b["scholarship"]))),
    "Age at enrollment": ("อายุตอนเข้า", lambda a, b: int(b["age_at_enrollment"])),
    "is_mature": ("อายุตอนเข้ามากกว่า 22", lambda a, b: "ใช่" if b["age_at_enrollment"] > 22
                  else "ไม่ใช่"),
    "Gender": ("เพศ", lambda a, b: "ชาย" if b["male"] else "หญิง"),
    "Marital Status": ("สถานภาพสมรส", lambda a, b: "โสด" if b["single"] else "อื่น ๆ"),
    "Displaced": ("ภูมิลำเนาต่างจังหวัด", lambda a, b: "ใช่" if not b["same_province"]
                  else "ไม่ใช่"),
    "Educational special needs": ("ความพิการ", lambda a, b: "มี" if b["disability"] else "ไม่มี"),
    "Mother's qualification": ("วุฒิมารดา", lambda a, b: b["mother_education"]),
    "Father's qualification": ("วุฒิบิดา", lambda a, b: b["father_education"]),
    "Mother's occupation": ("อาชีพมารดา", lambda a, b: b["mother_occupation"]),
    "Father's occupation": ("อาชีพบิดา", lambda a, b: b["father_occupation"]),
}


def factors_in_kku_terms(shap_row: pd.Series, a: pd.Series, b: pd.Series) -> tuple[pd.DataFrame, float]:
    """ค่า SHAP ของโมเดล UCI -> ตารางปัจจัยในภาษา มข. (เฉพาะที่มาจากข้อมูล มข. จริง)
    คืน (ตาราง, ผลรวม SHAP ของฟีเจอร์ที่ไม่มีใน มข. ซึ่งถูกเติมค่าคงที่/ค่ากลาง)."""
    rows, filled = [], 0.0
    for f, v in shap_row.items():
        if f in UCI_TO_KKU:
            name, get = UCI_TO_KKU[f]
            rows.append({"feature": f, "ปัจจัย": name, "ค่า": str(get(a, b)), "value": float(v)})
        else:
            filled += float(v)
    df = pd.DataFrame(rows)
    if len(df):
        df = df.reindex(df["value"].abs().sort_values(ascending=False).index)
    return df.reset_index(drop=True), filled


# ------------------------------------------------------------- ไทม์ไลน์รายภาค
def build_timeline(raw: pd.DataFrame, uci_model, uci_features: list[str], threshold: float,
                   train: pd.DataFrame, kku_bundle: dict | None = None):
    """คืน (timeline, term_features, uci_windows, source)
    source = 'kku' ถ้าใช้โมเดล มข., 'uci' ถ้าใช้โมเดล UCI ชั่วคราว."""
    tf = term_features(raw)
    win = uci_windows(tf, train)
    if kku_bundle is not None:
        source = "kku"
        thr = float(kku_bundle.get("threshold", threshold))
        tf["model_prob"] = kku_bundle["model"].predict_proba(tf[KKU_FEATURES])[:, 1]
    else:
        source = "uci"
        thr = threshold
        tf["model_prob"] = np.nan
        if len(win):
            p = uci_model.predict_proba(win[uci_features])[:, 1]
            key = dict(zip(zip(win[ID_COL], win["term_no"]), p))
            tf["model_prob"] = [key.get((s, t), np.nan)
                                for s, t in zip(tf["student_id"], tf["term_no"])]
    model_name = "โมเดล มข." if source == "kku" else "โมเดล (2 ภาคล่าสุด)"

    out = []
    for sid, g in tf.groupby("student_id", sort=False):
        g = g.reset_index(drop=True)
        for i, r in g.iterrows():
            t = int(r["term_no"])
            level, reasons = 0, []

            def raise_to(lv: int, why: str):
                nonlocal level
                level = max(level, lv)
                reasons.append(why)

            # ชั้น 1: โมเดล
            p = r["model_prob"]
            if not pd.isna(p):
                if p >= thr:
                    raise_to(2, f"{model_name} ประเมิน {p:.0%} เกินเกณฑ์ {thr:.3f}")
                elif p >= thr * WATCH_FACTOR:
                    raise_to(1, f"{model_name} ประเมิน {p:.0%} ใกล้เกณฑ์")
            # ชั้น 2: ข้อมูล มข.
            if r["term_result"] == "พ้นสภาพ":
                raise_to(3, "ทะเบียนระบุพ้นสภาพในภาคนี้")
            elif r["term_result"] == "รอพินิจ":
                raise_to(2, f"ทะเบียนระบุรอพินิจ (สะสม {int(r['terms_on_probation'])} ภาค)")
            if r["credits_behind"] >= CREDIT_BEHIND:
                raise_to(1, f"หน่วยกิตสะสมช้ากว่าแผน {int(r['credits_behind'])} หน่วยกิต")
            if r["n_f"] + r["n_w"] >= FAIL_W_LIMIT:
                raise_to(1, f"F {int(r['n_f'])} วิชา · W {int(r['n_w'])} วิชา")
            if r["outstanding_debt"]:
                raise_to(1, "ค้างชำระค่าธรรมเนียม")
            if r["late_fine"]:
                raise_to(1, "มีค่าปรับลงทะเบียนช้า")
            if not pd.isna(r["logins_change"]) and r["logins_change"] <= -LOGIN_DROP:
                raise_to(1, f"เข้าระบบทะเบียนลดลง {abs(r['logins_change']):.0%}")
            # ชั้น 3: แนวโน้มและเกณฑ์ GPAX
            gpax = float(r["gpax"])
            if t >= 2 and gpax < DISMISS_GPAX:
                raise_to(3, f"GPAX {gpax:.2f} ต่ำกว่า {DISMISS_GPAX:.2f} (เกณฑ์พ้นสภาพ)")
            elif t >= 2 and gpax < PROBATION_GPAX:
                raise_to(2, f"GPAX {gpax:.2f} ต่ำกว่า {PROBATION_GPAX:.2f} (เกณฑ์รอพินิจ)")
            elif gpax < WATCH_GPAX:
                raise_to(1, f"GPAX {gpax:.2f} ใกล้เกณฑ์รอพินิจ")
            if i >= 1:
                drop = r["gpa_change"]
                if drop <= -GPA_DROP:
                    raise_to(1, f"GPA ลดลง {abs(drop):.2f} จากภาคก่อน")
                    if r["gpa"] < PROBATION_GPAX:
                        raise_to(2, f"GPA ดิ่งจนต่ำกว่า {PROBATION_GPAX:.2f}")
                if i >= 2 and drop < 0 and g.at[i - 1, "gpa_change"] < 0:
                    raise_to(1, "GPA ลดลงต่อเนื่อง 3 ภาค")
                if level >= 1 and drop >= GPA_DROP:
                    reasons.append("แนวโน้มดีขึ้นชัดเจน")
            out.append({ID_COL: sid, "term_no": t, "ภาค": term_label(t),
                        "cr": int(r["cr"]), "cp": int(r["cp"]), "cp_cum": int(r["cp_cum"]),
                        "credits_behind": int(r["credits_behind"]),
                        "gpa": float(r["gpa"]), "gpax": gpax, "n_f": int(r["n_f"]),
                        "n_w": int(r["n_w"]), "term_result": r["term_result"] or "—",
                        "model_prob": p, "ระดับ": LEVELS[level],
                        "เหตุผล": " · ".join(reasons) if reasons else "—"})
    return pd.DataFrame(out), tf, win, source


def first_alert(timeline: pd.DataFrame, sid: str) -> str | None:
    """ภาคแรกที่ระดับขึ้นเป็น 'เสี่ยง' ขึ้นไป."""
    g = timeline[(timeline[ID_COL] == sid) & timeline["ระดับ"].isin(LEVELS[2:])]
    return None if g.empty else g.iloc[0]["ภาค"]


def is_known_at_risk(status: str) -> bool | None:
    if status in AT_RISK_STATUS:
        return True
    if status == "ปกติ":
        return False
    return None


# -------------------------------------------------------- label สำหรับโมเดล มข.
def make_labels(tf: pd.DataFrame, horizon: int = 2) -> pd.Series:
    """y ของภาค t = 1 ถ้าภาค t+1..t+horizon ในทะเบียนเป็นรอพินิจ/พ้นสภาพ
    = 0 ถ้ามีภาคถัดไปอย่างน้อย 1 ภาคและเป็นปกติทั้งหมด, NaN ถ้ายังไม่รู้อนาคต (ตัดทิ้งตอนฝึก)."""
    y = pd.Series(np.nan, index=tf.index)
    for _sid, g in tf.groupby("student_id", sort=False):
        res = g["term_result"].tolist()
        for k, idx in enumerate(g.index):
            if res[k] == "พ้นสภาพ":
                continue
            future = [x for x in res[k + 1:k + 1 + horizon] if x]
            if any(x in AT_RISK_STATUS for x in future):
                y[idx] = 1.0
            elif future:
                y[idx] = 0.0
    return y
