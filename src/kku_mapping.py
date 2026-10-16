"""
kku_mapping.py — แปลงข้อมูลจากระบบทะเบียน มข. ให้อยู่ในรูปแบบฟีเจอร์ของโมเดล (ชุด UCI)

ใช้ร่วมกันโดย app.py (ฟอร์ม/อัปโหลดในแดชบอร์ด) และ kku_real_sample.py
ไฟล์นี้ไม่มีข้อมูลส่วนบุคคล มีแค่กติกาการแปลง

หลักการแปลง (ดูตารางเทียบใน data_inventory_ตัวเอง.md):
  * ใช้ข้อมูลปีที่ 1 ภาค 1 และ 2 ให้ตรงกับช่วงเวลาที่ชุด UCI ใช้
  * GPA 0–4 -> คะแนน 0–20 ของ UCI (คูณ 5), เกรด ม.ปลาย 0–4 -> 0–200 (คูณ 50)
  * รหัสหมวดหมู่ของ UCI เป็นของโปรตุเกส จึงเลือกรหัสที่ความหมายใกล้เคียงที่สุด
  * ช่องที่ มข. ไม่มี (ลำดับการเลือก, คะแนนสอบเข้า, ตัวเลขเศรษฐกิจโปรตุเกส)
    เติมด้วยค่ามัธยฐานของชุดฝึก เพื่อไม่ให้ผลเอนไปทางใดทางหนึ่ง
"""
from __future__ import annotations

import math

import pandas as pd

from common import ID_COL, TARGET_COL
from data_prep import engineer_features

# ---------------------------------------------------------------- ตารางเทียบรหัส
PARENT_EDU = {              # วุฒิสูงสุดของบิดา/มารดา -> Mother's/Father's qualification
    "ประถมศึกษา": 37,        # Basic education 1st cycle
    "มัธยมศึกษาตอนต้น": 19,  # Basic education 3rd cycle
    "มัธยมศึกษาตอนปลาย": 1,  # Secondary education - 12th year
    "ปวช.": 1,
    "ปวส./อนุปริญญา": 39,    # Technological specialization course
    "ปริญญาตรี": 2,          # Higher education - bachelor's degree
    "ปริญญาโท": 4,
    "ปริญญาเอก": 5,
    "ไม่ทราบ": 34,           # Unknown
}
PARENT_OCC = {              # อาชีพหลักของบิดา/มารดา -> Mother's/Father's occupation
    "ค้าขาย/ธุรกิจส่วนตัว": 5,   # Personal services, security and safety workers and sellers
    "รับราชการ/รัฐวิสาหกิจ": 4,  # Administrative staff
    "พนักงานบริษัท": 4,
    "วิชาชีพ (แพทย์ วิศวกร ครู ฯลฯ)": 2,  # Specialists in intellectual and scientific activities
    "เกษตรกร": 6,                # Farmers and skilled workers in agriculture
    "รับจ้างทั่วไป": 9,           # Unskilled workers
    "ไม่ได้ทำงาน": 0,             # Student / not working
    "ไม่ทราบ": 90,                # Other situation
}
STATUS_OPTIONS = ["ปกติ", "รอพินิจ", "พ้นสภาพ", "ไม่ทราบ"]
AT_RISK_STATUS = {"รอพินิจ", "พ้นสภาพ"}

UCI_COURSE_CS = 9119         # Informatics Engineering (ใกล้วิทยาการคอมพิวเตอร์ที่สุด)
UCI_APP_MODE_GENERAL = 1     # 1st phase - general contingent (เทียบรับเข้าส่วนกลาง)
FILL_FROM_TRAIN = ("Application order", "Admission grade",
                   "Unemployment rate", "Inflation rate", "GDP")

# ---------------------------------------------------- คอลัมน์ไฟล์ CSV รูปแบบ มข.
KKU_COLUMNS = {
    "student_id": "รหัสสมมติ (ห้ามใช้ชื่อหรือรหัสจริง)",
    "status": "สถานะในทะเบียน: ปกติ / รอพินิจ / พ้นสภาพ / ไม่ทราบ",
    "sem1_courses_enrolled": "ปี 1 ภาค 1: จำนวนวิชาที่ลง",
    "sem1_courses_passed": "ปี 1 ภาค 1: จำนวนวิชาที่ผ่าน",
    "sem1_courses_w": "ปี 1 ภาค 1: จำนวนวิชาที่ถอน (W)",
    "sem1_gpa": "ปี 1 ภาค 1: GPA (0–4)",
    "sem2_courses_enrolled": "ปี 1 ภาค 2: จำนวนวิชาที่ลง",
    "sem2_courses_passed": "ปี 1 ภาค 2: จำนวนวิชาที่ผ่าน",
    "sem2_courses_w": "ปี 1 ภาค 2: จำนวนวิชาที่ถอน (W)",
    "sem2_gpa": "ปี 1 ภาค 2: GPA (0–4)",
    "hs_gpax": "เกรดเฉลี่ย ม.ปลาย (0–4)",
    "age_at_enrollment": "อายุตอนเข้าศึกษา (ปี)",
    "male": "เพศชาย 1 / หญิง 0",
    "single": "โสด 1 / อื่น ๆ 0",
    "same_province": "ภูมิลำเนาจังหวัดเดียวกับวิทยาเขต 1 / ต่างจังหวัด 0",
    "disability": "มีความพิการ 1 / ไม่มี 0",
    "father_education": "วุฒิบิดา (ตามรายการที่กำหนด)",
    "mother_education": "วุฒิมารดา (ตามรายการที่กำหนด)",
    "father_occupation": "อาชีพบิดา (ตามรายการที่กำหนด)",
    "mother_occupation": "อาชีพมารดา (ตามรายการที่กำหนด)",
    "has_outstanding_debt": "มีค่าธรรมเนียมค้างชำระ 1 / ไม่มี 0",
    "tuition_paid_on_time": "ชำระค่าเทอมตรงเวลา 1 / ไม่ตรง 0",
    "scholarship": "ได้รับทุน/กยศ. 1 / ไม่ได้ 0",
}


def template_df() -> pd.DataFrame:
    """ไฟล์ตัวอย่าง 1 แถวสำหรับกรอกข้อมูลรูปแบบ มข. (ค่าเป็นตัวอย่างกลาง ๆ ไม่ใช่ข้อมูลของใคร)"""
    return pd.DataFrame([{
        "student_id": "KKU-001", "status": "ปกติ",
        "sem1_courses_enrolled": 6, "sem1_courses_passed": 6, "sem1_courses_w": 0, "sem1_gpa": 3.00,
        "sem2_courses_enrolled": 7, "sem2_courses_passed": 7, "sem2_courses_w": 0, "sem2_gpa": 3.00,
        "hs_gpax": 3.00, "age_at_enrollment": 18, "male": 1, "single": 1,
        "same_province": 1, "disability": 0,
        "father_education": "ไม่ทราบ", "mother_education": "ไม่ทราบ",
        "father_occupation": "ไม่ทราบ", "mother_occupation": "ไม่ทราบ",
        "has_outstanding_debt": 0, "tuition_paid_on_time": 1, "scholarship": 0,
    }], columns=list(KKU_COLUMNS))


def validate_kku(kku: pd.DataFrame) -> list[str]:
    """ตรวจไฟล์/ข้อมูลรูปแบบ มข. คืนรายการปัญหา (ว่าง = ใช้ได้)."""
    missing = [c for c in KKU_COLUMNS if c not in kku.columns]
    if missing:
        return [f"ขาดคอลัมน์: {', '.join(missing)}"]
    errors = []
    for i, r in kku.reset_index(drop=True).iterrows():
        who = f"แถวที่ {i + 1} ({r['student_id']})"
        try:
            for s in (1, 2):
                enrolled = int(r[f"sem{s}_courses_enrolled"])
                passed = int(r[f"sem{s}_courses_passed"])
                w = int(r[f"sem{s}_courses_w"])
                if passed + w > enrolled:
                    errors.append(f"{who}: ภาค {s} วิชาที่ผ่าน + ถอน มากกว่าวิชาที่ลง")
                if not 0 <= float(r[f"sem{s}_gpa"]) <= 4:
                    errors.append(f"{who}: GPA ภาค {s} ต้องอยู่ระหว่าง 0–4")
            if not 0 <= float(r["hs_gpax"]) <= 4:
                errors.append(f"{who}: เกรด ม.ปลาย ต้องอยู่ระหว่าง 0–4")
        except (TypeError, ValueError):
            errors.append(f"{who}: มีช่องตัวเลขที่ว่างหรือไม่ใช่ตัวเลข")
        for col, table in (("father_education", PARENT_EDU), ("mother_education", PARENT_EDU),
                           ("father_occupation", PARENT_OCC), ("mother_occupation", PARENT_OCC)):
            if r[col] not in table:
                errors.append(f"{who}: '{r[col]}' ไม่อยู่ในรายการของ {col}")
        if r["status"] not in STATUS_OPTIONS:
            errors.append(f"{who}: สถานะต้องเป็น {' / '.join(STATUS_OPTIONS)}")
    return errors


def _to_uci_row(r) -> dict:
    row = {
        "Marital Status": 1 if int(r["single"]) else 2,
        "Application mode": UCI_APP_MODE_GENERAL,
        "Course": UCI_COURSE_CS,
        "Daytime/evening attendance": 1,
        "Previous qualification": 1,
        "Previous qualification (grade)": round(float(r["hs_gpax"]) * 50, 1),
        "Nationality": 1,
        "Mother's qualification": PARENT_EDU[r["mother_education"]],
        "Father's qualification": PARENT_EDU[r["father_education"]],
        "Mother's occupation": PARENT_OCC[r["mother_occupation"]],
        "Father's occupation": PARENT_OCC[r["father_occupation"]],
        "Displaced": 0 if int(r["same_province"]) else 1,
        "Educational special needs": int(r["disability"]),
        "Debtor": int(r["has_outstanding_debt"]),
        "Tuition fees up to date": int(r["tuition_paid_on_time"]),
        "Gender": int(r["male"]),
        "Scholarship holder": int(r["scholarship"]),
        "Age at enrollment": int(r["age_at_enrollment"]),
        "International": 0,
    }
    for s, tag in ((1, "1st"), (2, "2nd")):
        enrolled = int(r[f"sem{s}_courses_enrolled"])
        w = int(r[f"sem{s}_courses_w"])
        row[f"Curricular units {tag} sem (credited)"] = 0
        row[f"Curricular units {tag} sem (enrolled)"] = enrolled
        row[f"Curricular units {tag} sem (evaluations)"] = enrolled - w
        row[f"Curricular units {tag} sem (approved)"] = int(r[f"sem{s}_courses_passed"])
        row[f"Curricular units {tag} sem (grade)"] = round(float(r[f"sem{s}_gpa"]) * 5, 2)
        row[f"Curricular units {tag} sem (without evaluations)"] = w
    return row


def kku_to_uci(kku: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    """ข้อมูลรูปแบบ มข. -> DataFrame ที่มีฟีเจอร์ครบตามโมเดล + รหัส + label (ถ้าทราบสถานะ)."""
    kku = kku.reset_index(drop=True)
    df = pd.DataFrame([_to_uci_row(r) for _, r in kku.iterrows()])
    for col in FILL_FROM_TRAIN:
        df[col] = train[col].median()
    df = engineer_features(df)
    df.insert(0, ID_COL, kku["student_id"].astype(str).values)
    df[TARGET_COL] = [1.0 if s in AT_RISK_STATUS else (0.0 if s == "ปกติ" else math.nan)
                      for s in kku["status"]]
    return df
