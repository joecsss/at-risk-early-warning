"""
feature_labels.py — ชื่อฟีเจอร์ภาษาไทย, ตัวถอดรหัสค่าหมวดหมู่ และข้อเสนอแนะการช่วยเหลือ
ใช้ในแดชบอร์ดเพื่อให้อาจารย์ที่ปรึกษาอ่านผลได้โดยไม่ต้องรู้ชื่อคอลัมน์ต้นฉบับ
"""
from __future__ import annotations

# ---------------------------------------------------------------- ชื่อฟีเจอร์
FEATURE_LABELS_TH: dict[str, str] = {
    # ข้อมูลส่วนตัว / ภูมิหลัง
    "Marital Status": "สถานภาพสมรส",
    "Application mode": "ช่องทางการสมัครเข้าเรียน",
    "Application order": "ลำดับที่เลือกสาขานี้",
    "Course": "สาขาวิชา",
    "Daytime/evening attendance": "รอบเรียน (ปกติ/ค่ำ)",
    "Previous qualification": "วุฒิการศึกษาเดิม",
    "Previous qualification (grade)": "เกรดวุฒิการศึกษาเดิม",
    "Nationality": "สัญชาติ",
    "Mother's qualification": "วุฒิการศึกษาของมารดา",
    "Father's qualification": "วุฒิการศึกษาของบิดา",
    "Mother's occupation": "อาชีพของมารดา",
    "Father's occupation": "อาชีพของบิดา",
    "Admission grade": "คะแนนสอบเข้า",
    "Displaced": "ต้องย้ายถิ่นมาเรียน",
    "Educational special needs": "ความต้องการพิเศษทางการศึกษา",
    "Debtor": "มีหนี้ค้างชำระกับสถาบัน",
    "Tuition fees up to date": "ชำระค่าเล่าเรียนเป็นปัจจุบัน",
    "Gender": "เพศ",
    "Scholarship holder": "ได้รับทุนการศึกษา",
    "Age at enrollment": "อายุแรกเข้า",
    "International": "นักศึกษาต่างชาติ",
    # ผลการเรียนรายภาค
    "Curricular units 1st sem (credited)": "หน่วยกิตเทียบโอน ภาค 1",
    "Curricular units 1st sem (enrolled)": "วิชาที่ลงทะเบียน ภาค 1",
    "Curricular units 1st sem (evaluations)": "จำนวนการประเมิน/สอบ ภาค 1",
    "Curricular units 1st sem (approved)": "วิชาที่ผ่าน ภาค 1",
    "Curricular units 1st sem (grade)": "เกรดเฉลี่ย ภาค 1",
    "Curricular units 1st sem (without evaluations)": "วิชาที่ไม่ได้เข้าประเมิน ภาค 1",
    "Curricular units 2nd sem (credited)": "หน่วยกิตเทียบโอน ภาค 2",
    "Curricular units 2nd sem (enrolled)": "วิชาที่ลงทะเบียน ภาค 2",
    "Curricular units 2nd sem (evaluations)": "จำนวนการประเมิน/สอบ ภาค 2",
    "Curricular units 2nd sem (approved)": "วิชาที่ผ่าน ภาค 2",
    "Curricular units 2nd sem (grade)": "เกรดเฉลี่ย ภาค 2",
    "Curricular units 2nd sem (without evaluations)": "วิชาที่ไม่ได้เข้าประเมิน ภาค 2",
    # เศรษฐกิจมหภาค
    "Unemployment rate": "อัตราการว่างงาน (ภาวะเศรษฐกิจ)",
    "Inflation rate": "อัตราเงินเฟ้อ (ภาวะเศรษฐกิจ)",
    "GDP": "GDP (ภาวะเศรษฐกิจ)",
    # ฟีเจอร์ที่สร้างเพิ่ม (feature engineering)
    "pass_rate_1st": "สัดส่วนวิชาที่ผ่าน ภาค 1",
    "pass_rate_2nd": "สัดส่วนวิชาที่ผ่าน ภาค 2",
    "eval_rate_1st": "สัดส่วนวิชาที่เข้าประเมิน ภาค 1",
    "eval_rate_2nd": "สัดส่วนวิชาที่เข้าประเมิน ภาค 2",
    "grade_change": "ผลต่างเกรดเฉลี่ย (ภาค 2 − ภาค 1)",
    "pass_rate_change": "ผลต่างสัดส่วนวิชาที่ผ่าน (ภาค 2 − ภาค 1)",
    "financial_risk": "ความเสี่ยงด้านการเงิน (หนี้ค้าง/ค้างค่าเทอม/ไม่มีทุน)",
    "zero_pass_1st": "ไม่ผ่านสักวิชาในภาค 1",
    "no_grade_1st": "ไม่มีเกรดในภาค 1",
    "academic_gap": "ช่องว่างผลการเรียน (คะแนนสอบเข้า − เกรดภาค 1)",
    "is_mature": "นักศึกษาอายุมากกว่าเกณฑ์ปกติ (>22 ปี แรกเข้า)",
}

# ------------------------------------------------------- ตัวถอดรหัสค่าหมวดหมู่
YES_NO = {0: "ไม่ใช่", 1: "ใช่"}

VALUE_MAPS: dict[str, dict] = {
    "Gender": {0: "หญิง", 1: "ชาย"},
    "Daytime/evening attendance": {0: "รอบค่ำ", 1: "รอบปกติ"},
    "Marital Status": {
        1: "โสด", 2: "สมรส", 3: "หม้าย", 4: "หย่าร้าง",
        5: "อยู่กินฉันสามีภรรยา", 6: "แยกกันอยู่ตามกฎหมาย",
    },
    "Debtor": YES_NO,
    "Tuition fees up to date": YES_NO,
    "Scholarship holder": YES_NO,
    "Displaced": YES_NO,
    "International": YES_NO,
    "Educational special needs": YES_NO,
    "zero_pass_1st": YES_NO,
    "no_grade_1st": YES_NO,
    "is_mature": YES_NO,
    "Course": {
        33: "เทคโนโลยีการผลิตเชื้อเพลิงชีวภาพ",
        171: "ออกแบบแอนิเมชันและมัลติมีเดีย",
        8014: "สังคมสงเคราะห์ (รอบค่ำ)",
        9003: "เกษตรศาสตร์",
        9070: "ออกแบบนิเทศศิลป์",
        9085: "พยาบาลสัตว์",
        9119: "วิศวกรรมสารสนเทศ",
        9130: "การจัดการม้า",
        9147: "การจัดการ",
        9238: "สังคมสงเคราะห์",
        9254: "การท่องเที่ยว",
        9500: "พยาบาลศาสตร์",
        9556: "ทันตสาธารณสุข",
        9670: "การโฆษณาและการตลาด",
        9773: "วารสารศาสตร์และการสื่อสาร",
        9853: "การศึกษาขั้นพื้นฐาน",
        9991: "การจัดการ (รอบค่ำ)",
    },
}

# ------------------------------------------------- ข้อเสนอแนะการช่วยเหลือ (Phase 1)
# map: ฟีเจอร์ -> คำแนะนำเมื่อฟีเจอร์นั้นเป็น "ตัวดันความเสี่ยง" ของนักศึกษาคนนั้น
INTERVENTIONS: dict[str, str] = {
    "pass_rate_1st": "ผลการเรียนภาคแรกต่ำ — นัดพบเพื่อวางแผนการเรียน จัดติวเสริมรายวิชาที่ไม่ผ่าน",
    "pass_rate_2nd": "ผลการเรียนภาคสองต่ำ — ทบทวนภาระวิชาที่ลงทะเบียน พิจารณาลดหน่วยกิต",
    "Curricular units 1st sem (approved)": "ผ่านวิชาน้อยในภาค 1 — จัดพี่เลี้ยง/ติวเตอร์ และติดตามรายสัปดาห์",
    "Curricular units 2nd sem (approved)": "ผ่านวิชาน้อยในภาค 2 — ทบทวนแผนการเรียนร่วมกับอาจารย์ที่ปรึกษา",
    "Curricular units 1st sem (grade)": "เกรดเฉลี่ยภาค 1 ต่ำ — แนะนำคลินิกวิชาการ/ติวปรับพื้นฐาน",
    "Curricular units 2nd sem (grade)": "เกรดเฉลี่ยภาค 2 ต่ำ — ติดตามใกล้ชิดและประเมินความพร้อมก่อนลงทะเบียนภาคถัดไป",
    "grade_change": "ผลการเรียนมีแนวโน้มลดลง — ค้นหาสาเหตุ (สุขภาพ/การเงิน/ครอบครัว) โดยเร็ว",
    "pass_rate_change": "อัตราการผ่านวิชาลดลงจากภาคก่อน — สัญญาณเตือนให้เข้าพบทันที",
    "Curricular units 1st sem (without evaluations)": "ขาดสอบ/ไม่เข้าประเมินหลายวิชา — ตรวจสอบการเข้าเรียนและติดต่อผู้ปกครอง",
    "Curricular units 2nd sem (without evaluations)": "ขาดสอบ/ไม่เข้าประเมินหลายวิชา — ตรวจสอบการเข้าเรียนและติดต่อผู้ปกครอง",
    "eval_rate_1st": "เข้าประเมินไม่ครบ — ติดตามการเข้าชั้นเรียนและกำหนดการสอบ",
    "eval_rate_2nd": "เข้าประเมินไม่ครบ — ติดตามการเข้าชั้นเรียนและกำหนดการสอบ",
    "zero_pass_1st": "ไม่ผ่านสักวิชาในภาค 1 — เข้าข่ายเสี่ยงสูงมาก ควรพบอาจารย์ที่ปรึกษาโดยด่วน",
    "no_grade_1st": "ไม่มีเกรดในภาค 1 — ตรวจสอบว่าขาดเรียน ลาพัก หรือมีปัญหาการลงทะเบียน",
    "Debtor": "มีหนี้ค้างชำระ — ประสานงานฝ่ายกิจการนักศึกษาเรื่องผ่อนผัน/ทุนฉุกเฉิน",
    "Tuition fees up to date": "ค้างชำระค่าเล่าเรียน — แนะนำช่องทางผ่อนผันและกองทุนกู้ยืม",
    "financial_risk": "มีภาระทางการเงิน — แนะนำทุนการศึกษา/ทุนทำงาน (work-study)",
    "Scholarship holder": "ไม่ได้รับทุน — ตรวจสอบสิทธิ์และช่วยยื่นสมัครทุนที่เหมาะสม",
    "Admission grade": "พื้นฐานวิชาการแรกเข้าต่ำ — จัดโปรแกรมปรับพื้นฐานตั้งแต่ต้นภาค",
    "Previous qualification (grade)": "พื้นฐานการศึกษาเดิมต่ำ — ประเมินความพร้อมและเสริมทักษะพื้นฐาน",
    "academic_gap": "ผลการเรียนต่ำกว่าศักยภาพแรกเข้า — ค้นหาอุปสรรคที่ไม่ใช่ด้านวิชาการ",
    "Age at enrollment": "อายุแรกเข้าสูงกว่ากลุ่ม — อาจมีภาระงาน/ครอบครัว แนะนำแผนเรียนแบบยืดหยุ่น",
    "is_mature": "เป็นนักศึกษาผู้ใหญ่ — พิจารณาตารางเรียนที่ยืดหยุ่นและกลุ่มเพื่อนช่วยเพื่อน",
    "Displaced": "ย้ายถิ่นมาเรียน — ติดตามเรื่องที่พัก การปรับตัว และเครือข่ายทางสังคม",
    "International": "นักศึกษาต่างชาติ — สนับสนุนด้านภาษาและการปรับตัวทางวัฒนธรรม",
    "Educational special needs": "มีความต้องการพิเศษ — ประสานหน่วยบริการสนับสนุนนักศึกษา",
    "Application order": "สาขานี้ไม่ใช่อันดับต้นที่เลือก — พูดคุยเรื่องแรงจูงใจและเป้าหมายการเรียน",
    "Marital Status": "สถานภาพครอบครัวอาจมีผลต่อเวลาเรียน — สอบถามภาระนอกห้องเรียน",
    "Daytime/evening attendance": "เรียนรอบค่ำมักมีภาระงานควบคู่ — ตรวจสอบภาระงานและเวลาที่ใช้ทบทวน",
}

MACRO_FEATURES = {"Unemployment rate", "Inflation rate", "GDP"}

# ปัจจัยที่ "เปลี่ยนไม่ได้หรือไม่ควรใช้เป็นเหตุผลในการเรียกพบ" (ภูมิหลัง ครอบครัว ค่าคงที่ของชุด UCI)
# ยังแสดงในกราฟ SHAP ได้ตามจริง แต่ไม่นำมาสร้างข้อเสนอแนะ เพื่อไม่ให้เกิดการตีตรานักศึกษา
NON_ACTIONABLE = {
    # ชุด UCI
    "Gender", "Marital Status", "Nationality", "International", "Age at enrollment", "is_mature",
    "Mother's qualification", "Father's qualification", "Mother's occupation", "Father's occupation",
    "Educational special needs", "Displaced", "Course", "Application mode", "Application order",
    "Daytime/evening attendance", "Previous qualification", "Admission grade",
    "Previous qualification (grade)",
    "Curricular units 1st sem (enrolled)", "Curricular units 2nd sem (enrolled)",
    "Curricular units 1st sem (credited)", "Curricular units 2nd sem (credited)",
    *MACRO_FEATURES,
    # ชุด มข.
    "male", "age_at_enrollment", "father_education", "mother_education", "parents_marital",
    "siblings_studying", "same_province", "disability", "admission_method", "program_type",
    "term_no", "year", "hs_gpax",
}
BACKGROUND_NOTE = ("ปัจจัยด้านภูมิหลัง (เพศ อายุ ครอบครัว รอบรับเข้า ผลการเรียนก่อนเข้า) และจำนวนวิชาที่ลง "
                   "ไม่นำมาเป็นข้อเสนอแนะ เพราะเปลี่ยนไม่ได้และเสี่ยงต่อการตีตรานักศึกษา")


def is_actionable(feature: str) -> bool:
    """ปัจจัยนี้ใช้สร้างข้อเสนอแนะการช่วยเหลือได้ไหม."""
    return feature not in NON_ACTIONABLE


GENERIC_INTERVENTION = "นัดพบเพื่อพูดคุยและประเมินสถานการณ์รายบุคคล"

# ------------------------------------------------ เงื่อนไขก่อนสร้างข้อเสนอแนะ
# SHAP บวกแปลว่า "ค่านี้ทำให้โมเดลประเมินสูงกว่าคนเฉลี่ย" ไม่ได้แปลว่านักศึกษามีปัญหานั้นจริง
# (เช่น ไม่ผ่านสักวิชา = ไม่ใช่ แต่ SHAP บวกเล็กน้อยเพราะฟีเจอร์ซ้ำซ้อนกัน)
# จึงสร้างข้อเสนอแนะเมื่อผ่านครบ 4 ข้อ: ดันความเสี่ยงมากพอ, เป็นปัจจัยที่ช่วยได้,
# มีข้อเสนอแนะเฉพาะ และค่าจริงของนักศึกษาอยู่ฝั่งที่น่าห่วง
MIN_SUGGEST_SHAP = 0.20      # log-odds (odds เพิ่มราว 1.22 เท่า) ต่ำกว่านี้ถือว่าผลน้อยเกินไป
GRADE_LOW_UCI = 11.25        # เกรด UCI 0–20 = GPA มข. × 5 → เทียบ GPA ต่ำกว่า 2.25


def _sem(tag: str, what: str) -> str:
    return f"Curricular units {tag} sem ({what})"


CONCERN_UCI = {
    "zero_pass_1st": lambda r: r["zero_pass_1st"] == 1,
    "no_grade_1st": lambda r: r["no_grade_1st"] == 1,
    "Debtor": lambda r: r["Debtor"] == 1,
    "Tuition fees up to date": lambda r: r["Tuition fees up to date"] == 0,
    "Scholarship holder": lambda r: r["Scholarship holder"] == 0,
    "financial_risk": lambda r: r["financial_risk"] >= 1,
    "grade_change": lambda r: r["grade_change"] < 0,
    "pass_rate_change": lambda r: r["pass_rate_change"] < 0,
    "academic_gap": lambda r: r["academic_gap"] > 0,
    **{f"pass_rate_{t}": (lambda r, t=t: r[f"pass_rate_{t}"] < 1) for t in ("1st", "2nd")},
    **{f"eval_rate_{t}": (lambda r, t=t: r[f"eval_rate_{t}"] < 1) for t in ("1st", "2nd")},
    **{_sem(t, "approved"): (lambda r, t=t: r[_sem(t, "approved")] < r[_sem(t, "enrolled")])
       for t in ("1st", "2nd")},
    **{_sem(t, "grade"): (lambda r, t=t: r[_sem(t, "grade")] < GRADE_LOW_UCI)
       for t in ("1st", "2nd")},
    **{_sem(t, "without evaluations"): (lambda r, t=t: r[_sem(t, "without evaluations")] >= 1)
       for t in ("1st", "2nd")},
}


def should_suggest(feature: str, shap_value: float, row, rules: dict | None = None,
                   interventions: dict | None = None) -> bool:
    """สร้างข้อเสนอแนะจากปัจจัยนี้ไหม (row = ค่าฟีเจอร์ของนักศึกษาคนนั้น แบบ dict/Series)."""
    rules = CONCERN_UCI if rules is None else rules
    interventions = INTERVENTIONS if interventions is None else interventions
    if shap_value < MIN_SUGGEST_SHAP or not is_actionable(feature) or feature not in interventions:
        return False
    rule = rules.get(feature)
    if rule is None:
        return True
    try:
        return bool(rule(row))
    except (KeyError, TypeError, ValueError):
        return False


SUGGEST_NOTE = ("ข้อเสนอแนะสร้างเมื่อระดับไม่ใช่ 'ปกติ' จากปัจจัยที่ดันความเสี่ยงตั้งแต่ "
                f"{MIN_SUGGEST_SHAP:.2f} log-odds และค่าจริงของนักศึกษาอยู่ฝั่งที่น่าห่วง — "
                "ใช้ประกอบดุลยพินิจเสมอ · " + BACKGROUND_NOTE)


def label_th(feature: str) -> str:
    """ชื่อภาษาไทยของฟีเจอร์ (ถ้าไม่มีในตาราง คืนชื่อเดิม)."""
    return FEATURE_LABELS_TH.get(feature, feature)


def decode_value(feature: str, value):
    """แปลงค่ารหัสเป็นข้อความอ่านง่าย (ถ้ามีตัวถอดรหัส)."""
    mapping = VALUE_MAPS.get(feature)
    if not mapping:
        return value
    try:
        return mapping.get(int(value), value)
    except (TypeError, ValueError):
        return value


def suggest(feature: str) -> str:
    """ข้อเสนอแนะการช่วยเหลือสำหรับฟีเจอร์ที่ดันความเสี่ยง."""
    if feature in MACRO_FEATURES:
        return "ปัจจัยเศรษฐกิจระดับมหภาค — ไม่ใช่ปัจจัยรายบุคคล ใช้ประกอบบริบทเท่านั้น"
    return INTERVENTIONS.get(feature, GENERIC_INTERVENTION)
