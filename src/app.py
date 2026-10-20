"""
app.py — แดชบอร์ดสำหรับอาจารย์ที่ปรึกษา (Streamlit)

รัน:  streamlit run src/app.py

หน้าที่ของระบบ: "สนับสนุนการตัดสินใจ" ของอาจารย์ — จัดลำดับนักศึกษาที่ควรเข้าไปดูแลก่อน
พร้อมเหตุผลประกอบ (SHAP) ไม่ใช่การตัดสินผลการเรียนอัตโนมัติ
"""
from __future__ import annotations

import hmac
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import joblib
from common import (ID_COL, METRICS_JSON, MODEL_FILE, MODEL_KKU_FILE, TARGET_COL, TEST_CSV,
                    TRAIN_CSV)
from explain import RiskExplainer
from feature_labels import (INTERVENTIONS, SUGGEST_NOTE, decode_value, label_th,
                            should_suggest)
from kku_mapping import PARENT_EDU, PARENT_OCC, STATUS_OPTIONS
import db
from kku_schema import CHOICES, TERM_FIELDS, dictionary_df
from kku_schema import COLUMNS as KKU_SCHEMA_COLUMNS
from kku_schema import template_df as kku_template_df
from kku_schema import ensure_long, template_wide_df, to_wide
from kku_schema import validate as validate_kku_terms
from kku_timeline import (DISMISS_GPAX, KKU_CONCERN, KKU_FEATURE_LABELS, KKU_FEATURES,
                          KKU_INTERVENTIONS, LEVEL_COLORS, PROBATION_GPAX, UCI_TIPS_IN_KKU,
                          WATCH_FACTOR, build_timeline, factors_in_kku_terms, first_alert,
                          is_known_at_risk)

st.set_page_config(page_title="ระบบเตือนภัยล่วงหน้านักศึกษาเสี่ยง",
                   page_icon="🎓", layout="wide")

# ------------------------------------------------------------- ธีมและหน้าตา
# สีอิฐอ่านจาก .streamlit/config.toml (theme.primaryColor) — แก้ที่ไฟล์นั้นที่เดียว
BRICK = st.get_option("theme.primaryColor") or "#A3422B"
LINE = st.get_option("theme.borderColor") or "#DDD8D3"                 # เส้นขอบบาง
SURFACE = st.get_option("theme.secondaryBackgroundColor") or "#F4F2F0"  # เทาอ่อนมาก


def _shade(hex_color: str, factor: float) -> str:
    """ปรับสีให้เข้มขึ้น (factor < 1) หรืออ่อนลง (factor > 1) ใช้กับเส้นขอบ/แถบหัว"""
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return hex_color
    rgb = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return "#" + "".join(f"{min(255, max(0, round(c * factor))):02X}" for c in rgb)


BRICK_DARK = _shade(BRICK, 0.76)

# สีระดับความเสี่ยง — ความหมายเดิม ปรับโทนให้เข้ากับสีอิฐและอ่านออกบนพื้นขาว
BAND_COLORS = {"เสี่ยงสูง": "#9E2B1E", "เสี่ยงปานกลาง": "#C8781F",
               "เฝ้าระวัง": "#4A6B8A", "ปกติ": "#7D8A93"}
# ระดับของแท็บ มข. ใช้ชุดสีเดียวกัน โดยไม่แก้ kku_timeline.py ("เสี่ยง" = ปานกลาง)
_LEVEL_TO_BAND = {"ปกติ": "ปกติ", "เฝ้าระวัง": "เฝ้าระวัง",
                  "เสี่ยง": "เสี่ยงปานกลาง", "เสี่ยงสูง": "เสี่ยงสูง"}
LEVEL_PALETTE = {lv: BAND_COLORS.get(_LEVEL_TO_BAND.get(lv, ""), c)
                 for lv, c in LEVEL_COLORS.items()}
RISE_COLOR = BAND_COLORS["เสี่ยงสูง"]     # ปัจจัยที่เพิ่มความเสี่ยง
FALL_COLOR = "#2E7D55"                    # ปัจจัยที่ลดความเสี่ยง
STEEL = BAND_COLORS["เฝ้าระวัง"]          # แท่งกราฟข้อมูลทั่วไป
MUTED = "#9AA3A8"                         # เส้นอ้างอิง/เส้นตามแผน


def inject_style() -> None:
    """โหลดฟอนต์ Sarabun และใส่ CSS ของธีมทั้งหมด — เรียกครั้งเดียวตอนต้นแอป

    ใช้เฉพาะตัวเลือกที่คงที่ข้ามเวอร์ชัน (data-testid, data-baseweb, ชื่อ element)
    ไม่ผูกกับคลาสสุ่มของ Streamlit เช่น .css-1d391kg เพราะคลาสพวกนั้นเปลี่ยนทุกเวอร์ชัน
    """
    font_url = ("https://fonts.googleapis.com/css2?"
                "family=Sarabun:wght@400;500;600;700&display=swap")
    variables = (f":root{{--kku-brick:{BRICK};--kku-brick-dark:{BRICK_DARK};"
                 f"--kku-line:{LINE};--kku-surface:{SURFACE};"
                 "--kku-font:'Sarabun','Segoe UI',Tahoma,sans-serif;}")
    css = """
/* ---------- ฟอนต์ (ถ้าโหลด Google Fonts ไม่ได้ จะตกไปใช้ sans-serif ของเครื่อง) ---------- */
html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"],
button, input, select, textarea, [data-baseweb="select"] {
    font-family: var(--kku-font);
}

/* ---------- โครงหน้า ---------- */
[data-testid="stHeader"] { background: transparent; height: 2.6rem; }
[data-testid="stAppDeployButton"] { display: none; }
[data-testid="stMainBlockContainer"] { max-width: 1240px; padding: 2.9rem 2.5rem 1.5rem; }
[data-testid="stSidebar"] { background: #F7F6F4; border-right: 1px solid var(--kku-line); }
[data-testid="stSidebarUserContent"] { padding-top: 1.1rem; }

/* ---------- แถบหัวเว็บ ---------- */
.kku-header {
    background: var(--kku-brick);
    border-bottom: 4px solid var(--kku-brick-dark);
    border-radius: 4px;
    padding: 1.05rem 1.4rem;
    margin-bottom: 1.5rem;
}
.kku-header .kku-title {
    color: #FFFFFF; font-size: 1.4rem; font-weight: 600; line-height: 1.45;
}
.kku-header .kku-sub {
    color: #F0DCD6; font-size: .82rem; font-weight: 400; line-height: 1.55; margin-top: .3rem;
}

/* ---------- ส่วนท้าย ---------- */
.kku-footer {
    background: var(--kku-surface);
    border-top: 3px solid var(--kku-brick);
    border-radius: 4px;
    padding: .95rem 1.25rem;
    margin-top: 2.2rem;
    color: #45403C; font-size: .83rem; line-height: 1.6;
}
.kku-footer .kku-note { color: #6E6660; font-size: .78rem; margin-top: .35rem; }

/* ---------- หัวข้อ ---------- */
[data-testid="stMain"] h2 { font-size: 1.2rem; font-weight: 600; padding-top: .4rem; }
[data-testid="stMain"] h3 {
    font-size: 1.02rem; font-weight: 600;
    border-left: 3px solid var(--kku-brick); padding-left: .6rem;
}
[data-testid="stSidebar"] h2 { font-size: .98rem; font-weight: 600; }
[data-testid="stSidebar"] h3 { font-size: .92rem; font-weight: 600; }

/* ---------- การ์ดตัวเลขสรุป ---------- */
[data-testid="stColumn"] > [data-testid="stVerticalBlock"] { height: 100%; }
[data-testid="stElementContainer"]:has(> [data-testid="stMetric"]) { height: 100%; }
[data-testid="stMetric"] {
    background: #FFFFFF; border: 1px solid var(--kku-line); border-radius: 4px;
    padding: .8rem 1rem; box-shadow: none;
    height: 100%; min-height: 5.2rem;   /* ให้การ์ดในแถวเดียวกันสูงเท่ากัน */
}
[data-testid="stMetricLabel"] p { font-size: .8rem; font-weight: 500; color: #5A544F; }
[data-testid="stMetricValue"] { font-size: 1.5rem; font-weight: 600; }

/* ---------- แท็บ ---------- */
[data-baseweb="tab-list"] { gap: .15rem; border-bottom: 1px solid var(--kku-line); }
[data-testid="stTab"] { font-weight: 500; padding: .55rem .95rem; }
[data-testid="stTab"][aria-selected="true"] { color: var(--kku-brick); font-weight: 600; }

/* ---------- ตาราง (สีหัวตารางตั้งไว้ที่ .streamlit/config.toml) ---------- */
[data-testid="stDataFrame"] { border: 1px solid var(--kku-line); border-radius: 4px; }

/* ---------- ปุ่ม ---------- */
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primaryFormSubmit"] {
    border: 1px solid var(--kku-brick-dark); border-radius: 4px; font-weight: 600;
}
[data-testid="stBaseButton-secondary"], [data-testid="stBaseButton-secondaryFormSubmit"],
[data-testid="stDownloadButton"] button { border-radius: 4px; font-weight: 500; }

/* ---------- กล่องข้อความ / ส่วนขยาย / ฟอร์ม ---------- */
[data-testid="stAlert"], [data-testid="stExpander"], [data-testid="stForm"] { border-radius: 4px; }
[data-testid="stForm"] { border: 1px solid var(--kku-line); }

/* ---------- จอแคบ (มือถือ ~375px) ---------- */
@media (max-width: 640px) {
    [data-testid="stMainBlockContainer"] { padding: 2.4rem 1rem 1.2rem; }
    .kku-header { padding: .85rem 1rem; }
    .kku-header .kku-title { font-size: 1.05rem; }
    .kku-header .kku-sub { font-size: .75rem; }
    [data-testid="stMetricValue"] { font-size: 1.25rem; }
    [data-testid="stTab"] { padding: .5rem .6rem; font-size: .9rem; }
}
"""
    st.markdown(f"<style>@import url('{font_url}');{variables}{css}</style>",
                unsafe_allow_html=True)


def render_header() -> None:
    """แถบหัวเว็บสีอิฐ (ไม่ใส่ตราหรือโลโก้ของมหาวิทยาลัย)"""
    st.markdown(
        '<div class="kku-header">'
        '<div class="kku-title">ระบบเตือนภัยล่วงหน้านักศึกษาที่เสี่ยงด้านผลการเรียน</div>'
        '<div class="kku-sub">ต้นแบบโครงงานวิทยาการคอมพิวเตอร์ มหาวิทยาลัยขอนแก่น'
        ' · ไม่ใช่ระบบทางการของมหาวิทยาลัย</div>'
        '</div>', unsafe_allow_html=True)


def render_footer() -> None:
    """แถบท้ายหน้า เตือนขอบเขตการใช้งานผลทำนาย"""
    st.markdown(
        '<div class="kku-footer">'
        'ผลการทำนายเป็นสัญญาณเตือนเพื่อสนับสนุนการตัดสินใจของอาจารย์ที่ปรึกษา '
        'ไม่ใช่ข้อสรุปเกี่ยวกับตัวนักศึกษา'
        '<div class="kku-note">ผลทำนายอาจมีอคติจากข้อมูลในอดีต '
        'ควรใช้เพื่อจัดลำดับการเข้าไปช่วยเหลือเท่านั้น '
        'ไม่ควรใช้ตัดสิทธิ์หรือเปิดเผยต่อบุคคลที่ไม่เกี่ยวข้อง</div>'
        '</div>', unsafe_allow_html=True)


inject_style()


# --------------------------------------------------------------- โหลดทรัพยากร
@st.cache_resource(show_spinner="กำลังโหลดโมเดล...")
def load_model():
    if not os.path.exists(MODEL_FILE):
        st.error("ไม่พบไฟล์โมเดล — รัน `python src/data_prep.py` แล้วตามด้วย "
                 "`python src/train.py` ก่อนเปิดแดชบอร์ด")
        st.stop()
    return joblib.load(MODEL_FILE)


@st.cache_resource(show_spinner="กำลังเตรียมตัวอธิบายผล (SHAP)...")
def load_explainer(_model, features: tuple[str, ...]):
    train = pd.read_csv(TRAIN_CSV)
    return RiskExplainer(_model, list(features), train)


@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    if not os.path.exists(METRICS_JSON):
        return {}
    import json
    with open(METRICS_JSON, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_default_data() -> pd.DataFrame:
    return pd.read_csv(TEST_CSV)


@st.cache_data(show_spinner=False)
def load_train() -> pd.DataFrame:
    return pd.read_csv(TRAIN_CSV)


@st.cache_data(show_spinner="กำลังคำนวณความเสี่ยง...")
def score(df: pd.DataFrame, _model, features: tuple[str, ...]) -> np.ndarray:
    return _model.predict_proba(df[list(features)])[:, 1]


@st.cache_data(show_spinner="กำลังคำนวณปัจจัยสำคัญภาพรวม...")
def global_importance(_expl, df: pd.DataFrame, features: tuple[str, ...],
                      n: int = 300) -> pd.Series:
    sample = df[list(features)].sample(min(n, len(df)), random_state=42)
    return _expl.global_importance(sample, k=15)


def risk_band(p: float, t: float) -> str:
    if p >= t + (1 - t) / 2:
        return "เสี่ยงสูง"
    if p >= t:
        return "เสี่ยงปานกลาง"
    if p >= t / 2:
        return "เฝ้าระวัง"
    return "ปกติ"


def factor_chart(expl, row: pd.DataFrame, k: int = 8):
    """กราฟปัจจัย SHAP ของนักศึกษา 1 คน คืน (chart, factors)."""
    factors = expl.top_factors(row[features], k=k)
    fdf = pd.DataFrame(factors, columns=["feature", "value"])
    fdf["ปัจจัย"] = fdf["feature"].map(label_th)
    fdf["ค่าของนักศึกษา"] = [str(decode_value(f, row[f].iloc[0])) for f in fdf["feature"]]
    fdf["ทิศทาง"] = np.where(fdf["value"] > 0, "เพิ่มความเสี่ยง", "ลดความเสี่ยง")
    chart = (
        alt.Chart(fdf).mark_bar().encode(
            x=alt.X("value:Q", title=f"อิทธิพลต่อความเสี่ยง ({expl.units})"),
            y=alt.Y("ปัจจัย:N", sort=None, title=None),
            color=alt.Color("ทิศทาง:N", scale=alt.Scale(
                domain=["เพิ่มความเสี่ยง", "ลดความเสี่ยง"],
                range=[RISE_COLOR, FALL_COLOR]), legend=alt.Legend(title=None)),
            tooltip=["ปัจจัย", "ค่าของนักศึกษา",
                     alt.Tooltip("value:Q", title="ค่า SHAP", format="+.3f")],
        ).properties(height=320, width="container",
                     title="ปัจจัยที่มีผลต่อความเสี่ยงมากที่สุด (SHAP)")
    )
    return chart, factors


def collect_tips(fdf: pd.DataFrame, rule_row, rules=None, interventions=None,
                 wording=None, k: int = 4) -> list[tuple[str, str, float, str]]:
    """ตารางปัจจัย (feature, ปัจจัย, ค่า, value) -> [(ชื่อ, ค่า, SHAP, ข้อเสนอแนะ)]
    เฉพาะปัจจัยที่ผ่าน should_suggest (ดันความเสี่ยงมากพอ + ค่าจริงอยู่ฝั่งที่น่าห่วง)."""
    interventions = INTERVENTIONS if interventions is None else interventions
    wording = wording or {}
    tips = [(r.ปัจจัย, r.ค่า, r.value, wording.get(r.feature, interventions[r.feature]))
            for r in fdf.itertuples()
            if should_suggest(r.feature, r.value, rule_row, rules, interventions)]
    return tips[:k]


def show_tips(tips, level: str, heading: str = "ข้อเสนอแนะการช่วยเหลือ") -> None:
    """แสดงข้อเสนอแนะ — ระดับ 'ปกติ' ไม่สร้างข้อเสนอแนะ (กราฟ SHAP ยังแสดงเพื่อความโปร่งใส)."""
    st.subheader(heading)
    if level == "ปกติ":
        st.info("ระดับปกติ — ยังไม่มีข้อเสนอแนะ · กราฟด้านบนแสดงเหตุผลของโมเดลเพื่อความโปร่งใสเท่านั้น")
    elif tips:
        for name, val, v, tip in tips:
            st.markdown(f"- **{name}** (ค่าปัจจุบัน: {val}, อิทธิพล {v:+.2f}) → {tip}")
    else:
        st.success("ไม่พบปัจจัยด้านการเรียนหรือการเงินที่ชัดพอจะใช้เป็นข้อเสนอแนะ — "
                   "ดูเหตุผลของระดับประกอบ")
    st.caption(SUGGEST_NOTE)


def render_suggestions(factors, row: pd.DataFrame, band: str) -> None:
    fdf = pd.DataFrame([{"feature": f, "ปัจจัย": label_th(f),
                         "ค่า": decode_value(f, row[f].iloc[0]), "value": v} for f, v in factors])
    show_tips(collect_tips(fdf, row.iloc[0]), band)


# ------------------------------------------------- ฟังก์ชันช่วยของข้อมูล มข.
@st.cache_resource(show_spinner=False)
def load_kku_bundle():
    """โมเดลที่ฝึกจากข้อมูล มข. จริง (train_kku.py) — ถ้ายังไม่มีคืน None."""
    return joblib.load(MODEL_KKU_FILE) if os.path.exists(MODEL_KKU_FILE) else None


@st.cache_resource(show_spinner="กำลังเตรียมตัวอธิบายผลของโมเดล มข....")
def load_kku_explainer(_bundle):
    return RiskExplainer(_bundle["model"], KKU_FEATURES, _bundle["background"])


def kku_factor_chart(fdf: pd.DataFrame, units: str):
    fdf = fdf.assign(ทิศทาง=np.where(fdf["value"] > 0, "เพิ่มความเสี่ยง", "ลดความเสี่ยง"))
    return alt.Chart(fdf).mark_bar().encode(
        x=alt.X("value:Q", title=f"อิทธิพลต่อความเสี่ยง ({units})"),
        y=alt.Y("ปัจจัย:N", sort=None, title=None),
        color=alt.Color("ทิศทาง:N", scale=alt.Scale(
            domain=["เพิ่มความเสี่ยง", "ลดความเสี่ยง"], range=[RISE_COLOR, FALL_COLOR]),
            legend=alt.Legend(title=None)),
        tooltip=["ปัจจัย", "ค่า", alt.Tooltip("value:Q", title="ค่า SHAP", format="+.3f")],
    ).properties(height=300, width="container")


def yes_no(v) -> int:
    return int(v == "ใช่")


def term_risk_chart(tl_df: pd.DataFrame, title: str, y_title: str | None, height: int):
    """กราฟความเสี่ยงจากโมเดลรายภาค — ภาคแรกไม่มีค่าโมเดล (ต้องใช้ 2 ภาคติดกัน)
    จึงล็อกแกน x ให้แสดงทุกภาค และเขียนกำกับว่า "ใช้กฎ" แทนการหายไปจากกราฟ."""
    order = tl_df["ภาค"].tolist()
    x = alt.X("ภาค:N", sort=order, scale=alt.Scale(domain=order), title=None)
    bars = alt.Chart(tl_df).mark_bar().encode(
        x=x,
        y=alt.Y("model_prob:Q", scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%"), title=y_title),
        color=alt.Color("ระดับ:N", scale=alt.Scale(
            domain=list(LEVEL_PALETTE), range=list(LEVEL_PALETTE.values())),
            legend=alt.Legend(title=None, orient="bottom")),
        tooltip=["ภาค", "ระดับ", alt.Tooltip("model_prob:Q", format=".0%"), "เหตุผล"],
    )
    no_model = tl_df[tl_df["model_prob"].isna()].assign(note="ใช้กฎ")
    notes = alt.Chart(no_model).mark_text(fontSize=11, color=MUTED).transform_calculate(
        ypos="0.06").encode(x=x, y=alt.Y("ypos:Q"), text="note:N",
                            tooltip=["ภาค", "ระดับ", "เหตุผล"])
    return alt.layer(bars, notes).properties(height=height, width="container", title=title)


# ------------------------------------------------- มุมมองผู้ใช้งาน (2 บทบาท)
VIEW_ADVISOR = "อาจารย์ที่ปรึกษา"
VIEW_ADMIN = "ผู้ดูแลระบบ"
TAB_LIST = "รายชื่อนักศึกษาที่ควรติดตาม"
TAB_PERSON = "รายละเอียดรายบุคคล"
TAB_MODEL = "ภาพรวมโมเดล"
TAB_KKU = "ข้อมูลรูปแบบทะเบียน มข."


def admin_password() -> str | None:
    """รหัสผ่านผู้ดูแลระบบจาก .streamlit/secrets.toml — คืน None ถ้ายังไม่ได้ตั้งค่า

    ถ้าไม่มีไฟล์ secrets การอ่าน st.secrets จะโยน error (StreamlitSecretNotFoundError)
    จึงต้องห่อด้วย try/except เพื่อไม่ให้ทั้งแอปล่ม — ดูแบบไฟล์ที่ .streamlit/secrets.toml.example
    """
    try:
        pw = str(st.secrets["ADMIN_PASSWORD"]).strip()
    except Exception:
        return None
    return pw or None


def select_view() -> str:
    """ตัวเลือกมุมมองบนสุดของแถบข้าง — คืนมุมมองที่ใช้งานได้จริง

    มุมมองผู้ดูแลระบบต้องใส่รหัสผ่านให้ถูกก่อน สถานะเก็บใน st.session_state["admin_ok"]
    ถ้ายังไม่ผ่าน (หรือยังไม่ได้ตั้งรหัส) จะคืนมุมมองอาจารย์ที่ปรึกษาเสมอ
    """
    logged_in = bool(st.session_state.get("admin_ok"))
    choice = st.radio(
        "มุมมองการใช้งาน", [VIEW_ADVISOR, VIEW_ADMIN], index=1 if logged_in else 0,
        help="อาจารย์ที่ปรึกษา: เห็นรายชื่อ ความเสี่ยง เหตุผล และข้อเสนอแนะ · "
             "ผู้ดูแลระบบ: เห็นผลการเทรนและเครื่องมือปรับแต่งเพิ่ม (ต้องใส่รหัสผ่าน)")
    if choice == VIEW_ADVISOR:
        return VIEW_ADVISOR

    if logged_in:
        st.success("อยู่ในมุมมองผู้ดูแลระบบ")
        if st.button("ออกจากมุมมองผู้ดูแล"):
            st.session_state["admin_ok"] = False
            st.rerun()
        return VIEW_ADMIN

    pw = admin_password()
    if pw is None:
        st.warning("ยังไม่ได้ตั้งรหัสผ่านผู้ดูแลระบบ — คัดลอก `.streamlit/secrets.toml.example` "
                   "เป็น `.streamlit/secrets.toml` แล้วกำหนด `ADMIN_PASSWORD` · "
                   "ระหว่างนี้ใช้มุมมองอาจารย์ที่ปรึกษา")
        return VIEW_ADVISOR

    with st.form("admin_login", clear_on_submit=True):
        entered = st.text_input("รหัสผ่านผู้ดูแลระบบ", type="password")
        if st.form_submit_button("เข้าสู่มุมมองผู้ดูแล"):
            # encode ก่อนเทียบ เพราะ compare_digest ไม่รับ str ที่มีอักขระนอก ASCII (เช่น ภาษาไทย)
            if hmac.compare_digest(entered.encode("utf-8"), pw.encode("utf-8")):
                st.session_state["admin_ok"] = True
                st.rerun()
            else:
                st.error("รหัสผ่านไม่ถูกต้อง")
    return VIEW_ADVISOR


def render_sidebar_settings(is_admin: bool) -> tuple[pd.DataFrame, float]:
    """ส่วนตั้งค่าในแถบข้าง — คืน (ข้อมูลนักศึกษา, เกณฑ์ที่ใช้ตัดสินว่าเสี่ยง)

    มุมมองอาจารย์ที่ปรึกษาจะไม่เห็นตัวอัปโหลด CSV ฟีเจอร์ UCI, สไลเดอร์เกณฑ์ และข้อมูลโมเดล
    จึงใช้ชุดทดสอบและเกณฑ์ที่จูนไว้ใน bundle คงที่ (ตรรกะการคำนวณเหมือนกันทั้งสองมุมมอง)
    """
    if not is_admin:
        st.caption("ระบบใช้เกณฑ์เตือนที่ผู้ดูแลระบบจูนไว้แล้ว — "
                   "รายชื่อเรียงจากเสี่ยงมากไปน้อยให้อัตโนมัติ")
        return load_default_data(), default_threshold

    st.header("สำหรับผู้ดูแลระบบ")

    uploaded = st.file_uploader("อัปโหลดข้อมูลนักศึกษา (CSV)", type="csv",
                                help="ต้องมีคอลัมน์ฟีเจอร์เหมือนชุดข้อมูลที่ใช้เทรน")
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        missing = [c for c in features if c not in df.columns]
        if missing:
            st.error(f"ไฟล์ขาดคอลัมน์ {len(missing)} ตัว เช่น {missing[:3]} — "
                     "ใช้ข้อมูลชุดทดสอบแทน")
            df = load_default_data()
        else:
            st.success(f"ใช้ข้อมูลที่อัปโหลด ({len(df)} คน)")
    else:
        df = load_default_data()

    threshold = st.slider(
        "เกณฑ์ตัดสินว่า 'เสี่ยง' (probability)", 0.05, 0.95,
        value=float(round(default_threshold, 3)), step=0.001, format="%.3f",
        help="ลดค่านี้ = จับกลุ่มเสี่ยงได้ครบขึ้น (recall สูง) แต่แจ้งเตือนเกินจริงมากขึ้น · "
             "มีผลเฉพาะหน้าจอผู้ดูแลในรอบนี้ อาจารย์ที่ปรึกษายังใช้เกณฑ์ที่จูนไว้",
    )
    st.caption(f"ระดับที่เกณฑ์นี้: เฝ้าระวัง ≥ {threshold / 2:.3f} · เสี่ยง ≥ {threshold:.3f} · "
               f"เสี่ยงสูง ≥ {threshold + (1 - threshold) / 2:.3f} · "
               f"รายภาค มข.: เฝ้าระวังจากโมเดล ≥ {threshold * WATCH_FACTOR:.3f}")
    if abs(threshold - round(default_threshold, 3)) > 1e-9:
        st.warning(f"กำลังทดลองเกณฑ์ {threshold:.3f} (ค่าที่จูนไว้ {default_threshold:.3f}) — "
                   "ตัวเลขบนจอนี้จะต่างจากเล่มและจากที่อาจารย์เห็น")

    st.divider()
    st.subheader("ข้อมูลโมเดล")
    st.write(f"**อัลกอริทึม:** `{bundle.get('model_name', '-')}`")
    st.write(f"**ชุดฟีเจอร์:** `{bundle.get('feature_set', '-')}` ({len(features)} ตัว)")
    st.write(f"**threshold ที่จูนไว้:** `{default_threshold:.3f}`")
    tm = bundle.get("test_metrics", {})
    if tm:
        st.write(f"**ผลบนชุดทดสอบ:** recall `{tm.get('recall', 0):.3f}` · "
                 f"precision `{tm.get('precision', 0):.3f}` · "
                 f"PR-AUC `{tm.get('pr_auc', 0):.3f}`")
    st.caption("ปรับเกณฑ์ให้เหมาะกับกำลังดูแลของอาจารย์: "
               "ถ้าดูแลได้จำนวนจำกัด ให้เพิ่มเกณฑ์เพื่อโฟกัสรายที่เสี่ยงสุด")
    return df, threshold


# ------------------------------------------------------------------- เริ่มโหลด
bundle = load_model()
model, features = bundle["model"], bundle["features"]
default_threshold = float(bundle.get("threshold", 0.5))
metrics = load_metrics()

render_header()


def configure_database() -> None:
    """ใช้ PostgreSQL ออนไลน์ถ้าตั้ง DATABASE_URL ใน secrets ไว้ ไม่งั้นใช้ SQLite ในเครื่อง."""
    try:
        url = str(st.secrets["DATABASE_URL"]).strip()
    except Exception:
        url = ""
    db.configure(url or None)


configure_database()


LEVEL_RANK = {"เสี่ยงสูง": 3, "เสี่ยง": 2, "เฝ้าระวัง": 1, "ปกติ": 0}
LEVEL_ORDER = ["เสี่ยงสูง", "เสี่ยง", "เฝ้าระวัง", "ปกติ"]


@st.cache_data(ttl=60, show_spinner="กำลังโหลดนักศึกษาจากฐานข้อมูล...")
def load_db_students(db_key: str) -> pd.DataFrame:
    """นักศึกษารูปแบบทะเบียน มข. ทั้งหมดในฐานข้อมูล (db_key แยก cache ตามฐานข้อมูลที่ใช้)."""
    return db.load_students()


@st.cache_data(show_spinner="กำลังประเมินนักศึกษา มข. ทุกภาค...")
def kku_overview(data: pd.DataFrame, thr: float, model_key: str):
    """ประเมินทุกคนทุกภาค — model_key เปลี่ยนเมื่อมี/ไม่มีโมเดล มข. เพื่อให้คำนวณใหม่."""
    return build_timeline(data, model, features, thr, load_train(), load_kku_bundle())


kku_bundle = load_kku_bundle()
try:
    kku_db = load_db_students(db.describe())
    db_error = None
except Exception as e:  # ฐานข้อมูลต่อไม่ได้ → กลับไปใช้ข้อมูลตัวอย่าง UCI
    kku_db, db_error = pd.DataFrame(), str(e)
has_kku = len(kku_db) > 0

# ------------------------------------------------------------------- แถบข้างซ้าย
with st.sidebar:
    role = select_view()
    is_admin = role == VIEW_ADMIN
    use_kku = has_kku
    if has_kku and is_admin:
        use_kku = st.radio("แหล่งรายชื่อในแท็บแรก", ["ฐานข้อมูล มข.", "ข้อมูลตัวอย่าง UCI"],
                           help="อาจารย์ที่ปรึกษาเห็นฐานข้อมูล มข. เสมอเมื่อมีข้อมูล") == "ฐานข้อมูล มข."
    st.divider()
    if use_kku:
        st.info(f"**เวอร์ชันต้นแบบ** · รายชื่อมาจากฐานข้อมูลรูปแบบทะเบียน มข. "
                f"({kku_db['student_id'].nunique():,} คน) · ใช้รหัสสมมติ")
    else:
        st.info("**เวอร์ชันต้นแบบ** · รายชื่อเป็นข้อมูลตัวอย่างสาธารณะ UCI (รหัสสมมติ) · "
                "ข้อมูลที่กรอกในแท็บ มข. ไม่ถูกบันทึก เว้นแต่ผู้ดูแลระบบกดบันทึกลงฐานข้อมูล")
    df, threshold = render_sidebar_settings(is_admin)

if ID_COL not in df.columns:
    df = df.copy()
    df.insert(0, ID_COL, [f"S{i:04d}" for i in range(1, len(df) + 1)])

# นักศึกษา มข. ที่เพิ่มจากแท็บ "ข้อมูลจริง มข." (เก็บในหน่วยความจำของเซสชันเท่านั้น)
kku_rows = st.session_state.get("kku_rows")
if kku_rows is not None and len(kku_rows):
    df = pd.concat([df, kku_rows], ignore_index=True)

proba = score(df, model, tuple(features))
work = df.copy()
work["risk"] = proba
work["band"] = [risk_band(p, threshold) for p in proba]

# ----------------------------------------------------------------- ตัวชี้วัดรวม
if db_error:
    st.warning(f"เชื่อมต่อฐานข้อมูลไม่ได้ จึงแสดงข้อมูลตัวอย่างแทน ({db_error[:120]})")

if use_kku:
    kku_tl, kku_tf, kku_win, kku_src = kku_overview(
        kku_db, threshold, "kku" if kku_bundle is not None else "uci")
    kku_latest = kku_tl.groupby(ID_COL, sort=False).tail(1).copy()
    kku_latest["rank"] = kku_latest["ระดับ"].map(LEVEL_RANK)
    kku_latest = kku_latest.sort_values(["rank", "model_prob", "gpax"],
                                        ascending=[False, False, True], na_position="last")
    st.info(f"**ข้อมูลรูปแบบทะเบียน มข.** จากฐานข้อมูล ({db.describe()}) · "
            f"ประเมินซ้ำทุกภาค · โมเดล: {'โมเดล มข.' if kku_src == 'kku' else 'โมเดล UCI ชั่วคราว (2 ภาคล่าสุด)'}"
            " + เกณฑ์จากข้อมูลทะเบียน")
    n_lv = kku_latest["ระดับ"].value_counts()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("นักศึกษาในฐานข้อมูล", f"{len(kku_latest):,} คน")
    c2.metric("เสี่ยงสูง (ควรพบด่วน)", f"{int(n_lv.get('เสี่ยงสูง', 0)):,} คน")
    c3.metric("เสี่ยง", f"{int(n_lv.get('เสี่ยง', 0)):,} คน")
    c4.metric("เฝ้าระวัง", f"{int(n_lv.get('เฝ้าระวัง', 0)):,} คน")
else:
    st.info("**ข้อมูลตัวอย่างสาธารณะ** · รายชื่อด้านล่างมาจากชุดข้อมูล UCI "
            "(มหาวิทยาลัยในประเทศโปรตุเกส) ใช้สาธิตการทำงานของระบบ ไม่ใช่นักศึกษา มข."
            + ("" if has_kku else " · ยังไม่มีนักศึกษา มข. ในฐานข้อมูล "
               "(ผู้ดูแลระบบเพิ่มได้ที่แท็บข้อมูลรูปแบบทะเบียน มข.)"))
    n_total = len(work)
    n_risk = int((work["risk"] >= threshold).sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("นักศึกษาทั้งหมด", f"{n_total:,} คน")
    c2.metric("เข้าเกณฑ์เสี่ยง", f"{n_risk:,} คน", f"{n_risk / max(n_total, 1) * 100:.1f}%")
    c3.metric("เสี่ยงสูง (ควรพบด่วน)", f"{int((work['band'] == 'เสี่ยงสูง').sum()):,} คน")
    if is_admin and TARGET_COL in work.columns:
        y = work[TARGET_COL]
        pred = (work["risk"] >= threshold).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        rec = tp / max(int((y == 1).sum()), 1)
        prec = tp / max(int((pred == 1).sum()), 1)
        c4.metric("Recall / Precision ที่เกณฑ์นี้", f"{rec:.2f} / {prec:.2f}",
                  help="คำนวณจากข้อมูลชุดนี้ที่มีเฉลยจริง")
    else:
        c4.metric("ค่าความเสี่ยงเฉลี่ย", f"{work['risk'].mean() * 100:.1f}%")


# แท็บ "ภาพรวมโมเดล" เป็นผลการเทรน จึงมีเฉพาะมุมมองผู้ดูแลระบบ (อาจารย์เห็น 3 แท็บ)
tab_names = [TAB_LIST, TAB_PERSON] + ([TAB_MODEL] if is_admin else []) + [TAB_KKU]
tabs = dict(zip(tab_names, st.tabs(tab_names)))

# ------------------------------------------------------------ แท็บ 1: รายชื่อ
with tabs[TAB_LIST]:
    if use_kku:
        f1, f2 = st.columns([3, 3])
        lv_pick = f1.multiselect("ระดับล่าสุด", LEVEL_ORDER, default=LEVEL_ORDER)
        query = f2.text_input("ค้นหารหัสนักศึกษา", placeholder="เช่น KKU-001")
        view = kku_latest[kku_latest["ระดับ"].isin(lv_pick)] if lv_pick else kku_latest
        if query:
            view = view[view[ID_COL].astype(str).str.contains(query.strip(), case=False)]
        table = pd.DataFrame({
            "รหัสนักศึกษา": view[ID_COL],
            "ภาคล่าสุด": view["ภาค"],
            "ระดับล่าสุด": view["ระดับ"],
            "GPAX": view["gpax"],
            "หน่วยกิตสะสม": view["cp_cum"],
            "ช้ากว่าแผน": view["credits_behind"],
            "โมเดล (%)": (view["model_prob"] * 100).round(1),
            "เตือนครั้งแรก": [first_alert(kku_tl, s_) or "—" for s_ in view[ID_COL]],
            "เหตุผล": view["เหตุผล"],
        })
        st.write(f"พบ **{len(view):,}** คนตามเงื่อนไข (เรียงจากระดับล่าสุดที่รุนแรงที่สุด)")
        st.dataframe(
            table, hide_index=True, height=430,
            column_config={
                "GPAX": st.column_config.NumberColumn(format="%.2f"),
                "โมเดล (%)": st.column_config.ProgressColumn(
                    "โมเดล (%)", min_value=0, max_value=100, format="%.1f%%"),
                "เหตุผล": st.column_config.TextColumn(width="large")})
        st.download_button(
            "ดาวน์โหลดรายชื่อ (CSV)",
            table.to_csv(index=False).encode("utf-8-sig"),
            file_name="kku_students_to_follow.csv", mime="text/csv")
    else:
        st.caption("ข้อมูลตัวอย่างสาธารณะ (UCI) · ใช้สาธิตการทำงาน")
        f1, f2, f3 = st.columns([2, 2, 3])
        bands = f1.multiselect("ระดับความเสี่ยง", list(BAND_COLORS),
                               default=["เสี่ยงสูง", "เสี่ยงปานกลาง"])
        if "Course" in work.columns:
            codes = sorted(work["Course"].unique())
            chosen = f2.multiselect(
                "สาขาวิชา", codes, default=[],
                format_func=lambda c: f"{decode_value('Course', c)} ({c})")
        else:
            chosen = []
        query = f3.text_input("ค้นหารหัสนักศึกษา", placeholder="เช่น S0123")

        view = work[work["band"].isin(bands)] if bands else work
        if chosen:
            view = view[view["Course"].isin(chosen)]
        if query:
            view = view[view[ID_COL].astype(str).str.contains(query.strip(), case=False)]
        view = view.sort_values("risk", ascending=False)

        highlight_cols = [c for c in [
            "pass_rate_1st", "Curricular units 1st sem (approved)",
            "Curricular units 1st sem (grade)", "pass_rate_2nd",
            "Tuition fees up to date", "Debtor", "Scholarship holder",
            "Age at enrollment", "Course",
        ] if c in view.columns]

        table = pd.DataFrame({
            "รหัสนักศึกษา": view[ID_COL],
            "ความเสี่ยง (%)": (view["risk"] * 100).round(1),
            "ระดับ": view["band"],
        })
        for c in highlight_cols:
            col = view[c]
            table[label_th(c)] = (
                col.map(lambda v: decode_value(c, v))
                if c in ("Course", "Tuition fees up to date", "Debtor", "Scholarship holder")
                else col.round(2)
            )
        if TARGET_COL in view.columns:
            table["ผลจริง (ข้อมูลย้อนหลัง)"] = view[TARGET_COL].map({1: "หลุดจากระบบ", 0: "ยังเรียนอยู่"})

        st.write(f"พบ **{len(view):,}** คนตามเงื่อนไข (เรียงจากเสี่ยงมากไปน้อย)")
        st.dataframe(
            table, hide_index=True, height=430,
            column_config={"ความเสี่ยง (%)": st.column_config.ProgressColumn(
                "ความเสี่ยง (%)", min_value=0, max_value=100, format="%.1f%%")},
        )
        st.download_button(
            "ดาวน์โหลดรายชื่อ (CSV)",
            table.to_csv(index=False).encode("utf-8-sig"),
            file_name="at_risk_students.csv", mime="text/csv",
        )

# -------------------------------------------------------- แท็บ 2: รายบุคคล
with tabs[TAB_PERSON]:
    if use_kku:
        opts = kku_latest[ID_COL].tolist()
        lv_of = dict(zip(kku_latest[ID_COL], kku_latest["ระดับ"]))
        term_of = dict(zip(kku_latest[ID_COL], kku_latest["ภาค"]))
        sid = st.selectbox("เลือกนักศึกษา (เรียงตามระดับล่าสุด)", opts,
                           format_func=lambda s_: f"{s_} — {lv_of[s_]} ({term_of[s_]})")
        tl_one = kku_tl[kku_tl[ID_COL] == sid]
        tf_one = kku_tf[kku_tf["student_id"] == sid].reset_index(drop=True)
        cur = tl_one.iloc[-1]
        prof = tf_one.iloc[-1]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("ระดับล่าสุด", cur["ระดับ"], cur["ภาค"], delta_color="off")
        m2.metric("GPAX ล่าสุด", f"{cur['gpax']:.2f}")
        m3.metric("หน่วยกิตสะสม", f"{cur['cp_cum']} / {int(prof['required_credits'])}",
                  f"ช้ากว่าแผน {cur['credits_behind']}" if cur["credits_behind"] else "ตามแผน",
                  delta_color="inverse" if cur["credits_behind"] else "off")
        m4.metric("เตือนครั้งแรก", first_alert(kku_tl, sid) or "ยังไม่เคย")

        st.altair_chart(term_risk_chart(tl_one, "ระดับความเสี่ยงรายภาค (สี = ระดับรวม)",
                                        "ความเสี่ยงจากโมเดล", 220))
        st.caption("ภาคแรกยังไม่มีค่าจากโมเดล เพราะโมเดลต้องดู 2 ภาคติดกัน "
                   "ภาคนั้นจึงประเมินด้วยกฎ (GPAX, F/W, หน่วยกิต, การเงิน) อย่างเดียว")
        st.dataframe(
            tl_one[["ภาค", "gpa", "gpax", "n_f", "n_w", "term_result", "ระดับ", "เหตุผล"]].rename(
                columns={"gpa": "GPA", "gpax": "GPAX", "n_f": "F", "n_w": "W",
                         "term_result": "ผลการศึกษา (ทะเบียน)"}),
            hide_index=True,
            column_config={"GPA": st.column_config.NumberColumn(format="%.2f"),
                           "GPAX": st.column_config.NumberColumn(format="%.2f"),
                           "เหตุผล": st.column_config.TextColumn(width="large")})

        st.markdown(f"**ปัจจัยที่โมเดลให้น้ำหนักในภาคล่าสุด ({cur['ภาค']})**")
        tips = []
        if kku_src == "kku":
            kexpl = load_kku_explainer(kku_bundle)
            krow = tf_one.iloc[[-1]]
            facs = kexpl.top_factors(krow[KKU_FEATURES], k=8)
            fdf = pd.DataFrame([{"feature": f, "ปัจจัย": KKU_FEATURE_LABELS.get(f, f),
                                 "ค่า": str(krow[f].iloc[0]), "value": v} for f, v in facs])
            st.altair_chart(kku_factor_chart(fdf, kexpl.units))
            tips = collect_tips(fdf, tf_one.iloc[-1], KKU_CONCERN, KKU_INTERVENTIONS)
        else:
            wrow = kku_win[(kku_win[ID_COL] == sid) & (kku_win["term_no"] == cur["term_no"])]
            if wrow.empty:
                st.caption("ต้องมีอย่างน้อย 2 ภาคจึงจะใช้โมเดลได้")
            else:
                uexpl = load_explainer(model, tuple(features))
                sv = uexpl.shap_values(wrow[features]).iloc[0]
                fdf, filled = factors_in_kku_terms(sv, tf_one.iloc[-2], tf_one.iloc[-1])
                st.altair_chart(kku_factor_chart(fdf.head(8), uexpl.units))
                st.caption("แสดงเฉพาะปัจจัยที่มาจากข้อมูล มข. จริง · ฟีเจอร์ของชุด UCI ที่ มข. "
                           f"ไม่มีถูกเติมค่าคงที่ (รวมกันมีผล {filled:+.3f}) และไม่นำมาตีความ")
                tips = collect_tips(fdf, wrow[features].iloc[0], wording=UCI_TIPS_IN_KKU)
        show_tips(tips, cur["ระดับ"])
    else:
        st.caption("ข้อมูลตัวอย่างสาธารณะ (UCI) · ใช้สาธิตการทำงาน")
        ranked = work.sort_values("risk", ascending=False)
        options = ranked[ID_COL].tolist()
        sid = st.selectbox(
            "เลือกนักศึกษา (เรียงตามความเสี่ยง)", options,
            format_func=lambda s: (
                f"{s} — {ranked.loc[ranked[ID_COL] == s, 'risk'].iloc[0] * 100:.1f}% "
                f"({ranked.loc[ranked[ID_COL] == s, 'band'].iloc[0]})"),
        )
        row = work[work[ID_COL] == sid]
        p = float(row["risk"].iloc[0])
        band = row["band"].iloc[0]

        left, right = st.columns([1, 2])
        with left:
            st.metric("ระดับความเสี่ยง", f"{p * 100:.1f}%", band)
            st.progress(min(p, 1.0))
            st.caption(f"เกณฑ์ปัจจุบัน: {threshold:.3f} — "
                       f"{'เข้าเกณฑ์ต้องติดตาม' if p >= threshold else 'ยังไม่เข้าเกณฑ์'}")
            for c in ("Course", "Age at enrollment", "Tuition fees up to date",
                      "Debtor", "Scholarship holder"):
                if c in row.columns:
                    st.write(f"**{label_th(c)}:** {decode_value(c, row[c].iloc[0])}")

        with right:
            expl = load_explainer(model, tuple(features))
            chart, factors = factor_chart(expl, row)
            st.altair_chart(chart)

        render_suggestions(factors, row, band)

# ------------------------------------- แท็บ 3: ภาพรวมโมเดล (เฉพาะผู้ดูแลระบบ)
if TAB_MODEL in tabs:
    with tabs[TAB_MODEL]:
        col_a, col_b = st.columns([3, 2])
        with col_a:
            st.subheader("ปัจจัยสำคัญระดับภาพรวม (mean |SHAP|)")
            expl = load_explainer(model, tuple(features))
            imp = global_importance(expl, df, tuple(features))
            idf = pd.DataFrame({"ปัจจัย": [label_th(i) for i in imp.index],
                                "ความสำคัญ": imp.values})
            st.altair_chart(
                alt.Chart(idf).mark_bar(color=STEEL).encode(
                    x=alt.X("ความสำคัญ:Q", title=f"mean |SHAP| ({expl.units})"),
                    y=alt.Y("ปัจจัย:N", sort="-x", title=None),
                    tooltip=["ปัจจัย", alt.Tooltip("ความสำคัญ:Q", format=".3f")],
                ).properties(height=420, width="container"))

        with col_b:
            st.subheader("การกระจายของความเสี่ยง")
            dist = work[["risk", "band"]]
            if len(dist) > 5000:  # ข้อจำกัดจำนวนแถวของ Altair
                dist = dist.sample(5000, random_state=42)
            st.altair_chart(
                alt.Chart(dist).mark_bar().encode(
                    x=alt.X("risk:Q", bin=alt.Bin(maxbins=25),
                            title="ความน่าจะเป็นที่จะเสี่ยง"),
                    y=alt.Y("count():Q", title="จำนวนนักศึกษา"),
                    color=alt.Color("band:N", scale=alt.Scale(
                        domain=list(BAND_COLORS), range=list(BAND_COLORS.values())),
                        legend=alt.Legend(title="ระดับ")),
                ).properties(height=260, width="container"))

            if metrics:
                st.subheader("ผลเปรียบเทียบโมเดล (5-fold CV)")
                rows = [{"โมเดล": name, "PR-AUC": r["cv"]["pr_auc"],
                         "ROC-AUC": r["cv"]["roc_auc"], "Recall@0.5": r["cv"]["recall"]}
                        for name, r in metrics.get("cv_results", {}).items()]
                st.dataframe(pd.DataFrame(rows).sort_values("PR-AUC", ascending=False),
                             hide_index=True)

        labeled = work[work[TARGET_COL].notna()] if TARGET_COL in work.columns else work.iloc[0:0]
        if len(labeled):
            y = labeled[TARGET_COL].astype(int)
            pred = (labeled["risk"] >= threshold).astype(int)
            tn = int(((pred == 0) & (y == 0)).sum())
            fp = int(((pred == 1) & (y == 0)).sum())
            fn = int(((pred == 0) & (y == 1)).sum())
            tp = int(((pred == 1) & (y == 1)).sum())
            c1, c2 = st.columns([2, 3])
            with c1:
                st.subheader(f"ตารางความสับสน ที่เกณฑ์ {threshold:.3f}")
                st.dataframe(pd.DataFrame(
                    [[f"TN = {tn}", f"FP = {fp} (เตือนเกิน)"],
                     [f"FN = {fn} (พลาด)", f"TP = {tp}"]],
                    index=["จริง: ไม่เสี่ยง", "จริง: เสี่ยง"],
                    columns=["ทาย: ไม่เสี่ยง", "ทาย: เสี่ยง"]))
                m1, m2, m3 = st.columns(3)
                m1.metric("Recall", f"{tp / max(tp + fn, 1):.2f}")
                m2.metric("Precision", f"{tp / max(tp + fp, 1):.2f}")
                m3.metric("Accuracy", f"{(tp + tn) / max(len(y), 1):.2f}")
                st.caption("ลดเกณฑ์ → FN น้อยลง (พลาดน้อย) แต่ FP มากขึ้น (เตือนเกิน)")
            with c2:
                st.subheader("ถ้าระบบบอกระดับนี้ เสี่ยงจริงกี่ %")
                order = list(BAND_COLORS)
                bt = (labeled.assign(y=y).groupby("band")["y"]
                      .agg(["size", "sum"]).reindex(order).fillna(0))
                st.dataframe(pd.DataFrame({
                    "ระดับที่ระบบแจ้ง": order,
                    "จำนวน": bt["size"].astype(int).values,
                    "เสี่ยงจริง": bt["sum"].astype(int).values,
                    "P(เสี่ยงจริง | ระดับ)": [100 * s / n if n else None
                                              for s, n in zip(bt["sum"], bt["size"])],
                }), hide_index=True, column_config={
                    "P(เสี่ยงจริง | ระดับ)": st.column_config.ProgressColumn(
                        min_value=0, max_value=100, format="%.1f%%")})
                st.caption(f"ภาพรวมทั้งกลุ่มเสี่ยงจริง {y.mean():.0%} · "
                           "ระดับที่สูงกว่าควรมีสัดส่วนเสี่ยงจริงสูงกว่า จึงใช้จัดลำดับการติดตามได้")

# --------------------------------------------------- แท็บ 4: ข้อมูลจริง มข.
with tabs[TAB_KKU]:
    kku_bundle = load_kku_bundle()
    st.subheader("ข้อมูลตามระบบทะเบียน มข. — ประเมินทุกภาค")
    if kku_bundle is None:
        st.caption(
            "ใช้ข้อมูลตามทะเบียน มข. (CR, CP, GPA, GPAX, F, W, ผลการศึกษารายภาค, ค้างชำระ, ค่าปรับ, "
            "รอบรับเข้า ฯลฯ) · กฎเตือนภัยใช้ข้อมูล มข. ได้ทันที · ส่วนโมเดลยังเป็นโมเดล UCI ชั่วคราว "
            "(แปลง 2 ภาคล่าสุดเข้าไป) จนกว่าจะฝึกโมเดล มข. ด้วย `python src/train_kku.py --data ...` · "
            "ข้อมูลอยู่ในหน่วยความจำของเซสชันนี้เท่านั้น")
    else:
        st.success(
            f"ใช้โมเดล มข. ที่ฝึกจากข้อมูลทะเบียน {kku_bundle['n_students']} คน "
            f"({kku_bundle['n_rows']} ตัวอย่างคน×ภาค) · ทำนายล่วงหน้า {kku_bundle['horizon']} ภาค · "
            f"PR-AUC (CV) {kku_bundle['cv_metrics']['pr_auc']:.3f}")

    with st.expander("ข้อมูล มข. ที่ระบบใช้ (พจนานุกรมข้อมูล)"):
        ddf = dictionary_df()
        st.dataframe(ddf, hide_index=True)
        st.caption(f"ทั้งหมด {len(ddf)} ฟิลด์ · เข้าโมเดล UCI ได้ {int((ddf['โมเดล UCI'] == '✓').sum())} · "
                   f"ใช้ในกฎเตือนภัย {int((ddf['กฎเตือนภัย'] == '✓').sum())} · "
                   f"ใช้ในโมเดล มข. {int((ddf['โมเดล มข.'] == '✓').sum())}")

    mode = st.radio("วิธีนำเข้า", ["กรอกทีละคน", "อัปโหลดไฟล์ CSV รูปแบบทะเบียน",
                                   "ข้อมูลที่บันทึกในฐานข้อมูล"], horizontal=True)

    if mode == "กรอกทีละคน":
        tpl = kku_template_df()
        with st.form("kku_form"):
            a, b = st.columns(2)
            sid_in = a.text_input("รหัส (ใช้รหัสสมมติ ห้ามใส่ชื่อหรือรหัสจริง)", "KKU-001")
            status_in = b.selectbox("สถานภาพปัจจุบันในทะเบียน", STATUS_OPTIONS)

            st.markdown("**ผลการเรียนรายภาค** — หน้าผลการเรียน + ภาระค่าใช้จ่าย "
                        "(เพิ่มแถวได้ทุกภาคที่เรียนแล้ว ไม่นับภาคฤดูร้อน)")
            ed = tpl[list(TERM_FIELDS)].copy()
            ed["outstanding_debt"] = ed["outstanding_debt"].astype(bool)
            ed["late_fine"] = ed["late_fine"].astype(bool)
            terms_in = st.data_editor(
                ed, num_rows="dynamic", hide_index=True, key="kku_terms_editor",
                column_config={
                    "term_no": st.column_config.NumberColumn("ลำดับภาค", min_value=1, step=1,
                                                             help=TERM_FIELDS["term_no"][0]),
                    "term": st.column_config.TextColumn("ภาค/ปี", help="เช่น 1/2566"),
                    "n_courses": st.column_config.NumberColumn("วิชาที่ลง", min_value=0, step=1),
                    "n_f": st.column_config.NumberColumn("F", min_value=0, step=1),
                    "n_w": st.column_config.NumberColumn("W", min_value=0, step=1),
                    "cr": st.column_config.NumberColumn("CR", min_value=0, step=1,
                                                        help=TERM_FIELDS["cr"][0]),
                    "cp": st.column_config.NumberColumn("CP", min_value=0, step=1,
                                                        help=TERM_FIELDS["cp"][0]),
                    "gpa": st.column_config.NumberColumn("GPA", min_value=0.0, max_value=4.0,
                                                         step=0.01, format="%.2f"),
                    "gpax": st.column_config.NumberColumn("GPAX", min_value=0.0, max_value=4.0,
                                                          step=0.01, format="%.2f",
                                                          help=TERM_FIELDS["gpax"][0]),
                    "term_result": st.column_config.SelectboxColumn(
                        "ผลการศึกษา", options=CHOICES["term_result"]),
                    "outstanding_debt": st.column_config.CheckboxColumn("ค้างชำระ"),
                    "late_fine": st.column_config.CheckboxColumn("ค่าปรับช้า"),
                    "logins": st.column_config.NumberColumn("เข้าระบบ (ครั้ง)", min_value=0, step=1),
                })
            st.caption("ค่าในตารางเป็นตัวอย่าง — แก้ให้ตรงกับหน้าผลการเรียน")

            st.markdown("**การศึกษา** (ระเบียนประวัติ / ตรวจสอบจบ)")
            c1, c2, c3 = st.columns(3)
            faculty = c1.text_input("คณะ", "วิทยาลัยการคอมพิวเตอร์")
            program = c2.text_input("หลักสูตร/สาขา", "วิทยาการคอมพิวเตอร์")
            program_type = c3.selectbox("ประเภท", CHOICES["program_type"])
            c1, c2, c3 = st.columns(3)
            admission = c1.selectbox("วิธีรับเข้า", CHOICES["admission_method"],
                                     index=CHOICES["admission_method"].index("ไม่ทราบ"))
            entry_year = c2.number_input("ปีที่เข้า (พ.ศ.)", 2550, 2600, 2566)
            req_credits = c3.number_input("หน่วยกิตตลอดหลักสูตร", 60, 250, 120)

            st.markdown("**ภูมิหลัง**")
            c1, c2, c3, c4 = st.columns(4)
            hs_gpax = c1.number_input("เกรดเฉลี่ย ม.ปลาย", 0.0, 4.0, 3.00, step=0.01)
            age = c2.number_input("อายุตอนเข้าศึกษา", 15, 60, 18)
            gender = c3.selectbox("เพศ", ["ชาย", "หญิง"])
            marital = c4.selectbox("สถานภาพสมรส", ["โสด", "สมรส/อื่น ๆ"])
            c1, c2, c3, c4 = st.columns(4)
            province = c1.selectbox("ภูมิลำเนาจังหวัดเดียวกับวิทยาเขต", ["ใช่", "ไม่ใช่"])
            disability = c2.selectbox("ความพิการ", ["ไม่ใช่", "ใช่"])
            health = c3.selectbox("มีโรคประจำตัว/ปัญหาสุขภาพ", ["ไม่ใช่", "ใช่"])
            dorm = c4.selectbox("ที่พัก", CHOICES["dorm"], index=CHOICES["dorm"].index("ไม่ทราบ"))

            st.markdown("**ครอบครัว**")
            c1, c2, c3 = st.columns(3)
            f_edu = c1.selectbox("วุฒิสูงสุดของบิดา", list(PARENT_EDU), index=list(PARENT_EDU).index("ไม่ทราบ"))
            m_edu = c2.selectbox("วุฒิสูงสุดของมารดา", list(PARENT_EDU), index=list(PARENT_EDU).index("ไม่ทราบ"))
            p_mar = c3.selectbox("สถานภาพบิดามารดา", CHOICES["parents_marital"],
                                 index=CHOICES["parents_marital"].index("ไม่ทราบ"))
            f_occ = c1.selectbox("อาชีพหลักของบิดา", list(PARENT_OCC), index=list(PARENT_OCC).index("ไม่ทราบ"))
            m_occ = c2.selectbox("อาชีพหลักของมารดา", list(PARENT_OCC), index=list(PARENT_OCC).index("ไม่ทราบ"))
            income = c3.selectbox("รายได้ครอบครัวต่อเดือน (บาท)", CHOICES["family_income"],
                                  index=CHOICES["family_income"].index("ไม่ทราบ"))
            siblings = c1.number_input("พี่น้องที่กำลังศึกษา", 0, 10, 0)

            st.markdown("**การเงิน**")
            c1, c2 = st.columns(2)
            schol = c1.selectbox("ได้รับทุน / กยศ.", ["ไม่ใช่", "ใช่"])
            funding = c2.selectbox("แหล่งเงินหลักที่ใช้เรียน", CHOICES["funding_source"],
                                   index=CHOICES["funding_source"].index("ไม่ทราบ"))

            if st.form_submit_button("ประเมินความเสี่ยงทุกภาค", type="primary"):
                profile = {
                    "student_id": sid_in.strip() or "KKU-001", "status": status_in,
                    "faculty": faculty, "program": program, "program_type": program_type,
                    "admission_method": admission, "entry_year": entry_year,
                    "required_credits": req_credits, "plan_terms": 8,
                    "hs_gpax": hs_gpax, "age_at_enrollment": age, "male": int(gender == "ชาย"),
                    "single": int(marital == "โสด"), "same_province": yes_no(province),
                    "disability": yes_no(disability), "health_issue": yes_no(health),
                    "dorm": dorm, "father_education": f_edu, "mother_education": m_edu,
                    "father_occupation": f_occ, "mother_occupation": m_occ,
                    "parents_marital": p_mar, "family_income": income,
                    "siblings_studying": siblings, "funding_source": funding,
                    "scholarship": yes_no(schol),
                }
                terms = terms_in.dropna(subset=["gpa"]).copy()
                for c in ("outstanding_debt", "late_fine"):
                    terms[c] = terms[c].fillna(False).astype(int)
                st.session_state["kku_preview"] = terms.assign(**profile)[KKU_SCHEMA_COLUMNS]

    elif mode == "อัปโหลดไฟล์ CSV รูปแบบทะเบียน":
        d1, d2 = st.columns(2)
        d1.download_button(
            "ไฟล์ตัวอย่าง: 1 คน = 1 แถว (แนะนำ)",
            template_wide_df().to_csv(index=False).encode("utf-8-sig"),
            file_name="kku_template_1row_per_student.csv", mime="text/csv",
            help="คอลัมน์รายเทอมตั้งชื่อ t<ลำดับเทอม>_<ฟิลด์> เช่น t1_gpa, t1_cr, t2_gpa · "
                 "เทอมที่ยังไม่ได้เรียนเว้นว่าง")
        d2.download_button(
            "ไฟล์ตัวอย่าง: 1 แถว = 1 คน × 1 เทอม",
            kku_template_df().to_csv(index=False).encode("utf-8-sig"),
            file_name="kku_template_1row_per_term.csv", mime="text/csv")
        st.caption("อัปโหลดได้ทั้ง 2 แบบ ระบบแยกเป็นรายเทอมให้เองเพราะต้องประเมินทีละเทอม · "
                   "แบบ 1 คน 1 แถว: เทอมที่ 1 คือ t1_..., เทอมที่ 2 คือ t2_... ไม่ต้องกรอกเทอมที่ยังไม่ได้เรียน · "
                   "คอลัมน์บังคับดูได้จากพจนานุกรมข้อมูลด้านบน คอลัมน์อื่นเว้นว่างได้ · ค่าที่เป็นตัวเลือก: "
                   + " | ".join(f"{k}: {' / '.join(o for o in v if o)}"
                                for k, v in CHOICES.items()
                                if k in ("term_result", "admission_method", "family_income")))
        up = st.file_uploader("อัปโหลดไฟล์ CSV", type="csv", key="kku_upload")
        if up is not None:
            st.session_state["kku_preview"] = ensure_long(pd.read_csv(up))

    else:
        try:
            saved = db.list_students()
        except Exception as e:  # ฐานข้อมูลเสียหรือเปิดไม่ได้ ต้องไม่ทำให้ทั้งแอปล่ม
            saved = None
            st.error(f"เปิดฐานข้อมูลไม่ได้: {e}")
        if saved is not None and saved.empty:
            st.info("ยังไม่มีข้อมูลในฐานข้อมูล · กรอกหรืออัปโหลด แล้วกด \"บันทึกลงฐานข้อมูล\" ด้านล่างผลประเมิน")
        elif saved is not None:
            st.dataframe(saved.rename(columns={
                "student_id": "รหัส", "status": "สถานภาพ", "source": "ที่มา",
                "consent": "ยินยอม", "updated_at": "แก้ไขล่าสุด", "n_terms": "จำนวนภาค",
                "last_term": "ภาคล่าสุด", "last_level": "ระดับที่ประเมินล่าสุด"}),
                hide_index=True)
            pick = st.multiselect("เลือกนักศึกษา", saved["student_id"].tolist(),
                                  default=saved["student_id"].tolist()[:1])
            c1, c2, c3 = st.columns(3)
            if c1.button("เปิดประเมิน", type="primary", disabled=not pick):
                st.session_state["kku_preview"] = db.load_students(pick)
                st.rerun()
            if is_admin:
                c2.download_button(
                    "ส่งออกทั้งหมด (CSV 1 คน 1 แถว)",
                    to_wide(db.load_students()).to_csv(index=False).encode("utf-8-sig"),
                    file_name="kku_database_export.csv", mime="text/csv")
                if c3.button("ลบที่เลือกออกจากฐานข้อมูล", disabled=not pick):
                    for sid in pick:
                        db.delete_student(sid)
                    load_db_students.clear()
                    st.session_state.pop("kku_preview", None)
                    st.rerun()
            if pick:
                with st.expander("ประวัติการประเมินที่บันทึกไว้"):
                    hist = pd.concat([db.assessment_history(sid) for sid in pick],
                                     ignore_index=True)
                    st.dataframe(hist.rename(columns={
                        "student_id": "รหัส", "term_no": "ภาค", "risk_level": "ระดับ",
                        "risk_score": "ความน่าจะเป็น", "model_source": "โมเดล",
                        "threshold": "เกณฑ์", "reasons": "เหตุผล",
                        "assessed_at": "ประเมินเมื่อ"}), hide_index=True)
        db_note = ("ข้อมูลอยู่ถาวรบนฐานข้อมูลออนไลน์" if db.is_online()
                   else "บนเว็บเดโม ไฟล์ SQLite จะหายเมื่อแอปรีสตาร์ท")
        st.caption(f"ฐานข้อมูล: {db.describe()} · ตาราง STUDENT, TERM_RECORD, RISK_ASSESSMENT "
                   f"ตามแผนภาพ ER · {db_note}"
                   + ("" if is_admin else " · การบันทึก ส่งออก และลบ ทำได้ในมุมมองผู้ดูแลระบบ"))


    preview = st.session_state.get("kku_preview")
    if preview is not None and len(preview):
        errors = validate_kku_terms(preview)
        if errors:
            st.error("ข้อมูลยังไม่ถูกต้อง:\n\n- " + "\n- ".join(errors[:15]))
        else:
            timeline, tfeat, windows, source = build_timeline(
                preview, model, features, threshold, load_train(), kku_bundle)
            st.divider()

            last = timeline.groupby(ID_COL, sort=False).tail(1)
            status_of = dict(zip(tfeat["student_id"], tfeat["status"]))
            if len(last) > 1:
                st.markdown("**สรุปรายคน**")
                st.dataframe(pd.DataFrame({
                    "รหัส": last[ID_COL], "ภาคล่าสุด": last["ภาค"], "GPAX": last["gpax"],
                    "หน่วยกิตสะสม": last["cp_cum"], "ระดับล่าสุด": last["ระดับ"],
                    "เตือนครั้งแรก": [first_alert(timeline, s) or "—" for s in last[ID_COL]],
                    "สถานภาพจริง": [status_of.get(s, "ไม่ทราบ") for s in last[ID_COL]],
                }), hide_index=True)

            sids = last[ID_COL].tolist()
            sid_sel = sids[0] if len(sids) == 1 else st.selectbox("ดูรายละเอียดของ", sids)
            tl = timeline[timeline[ID_COL] == sid_sel]
            tfs = tfeat[tfeat["student_id"] == sid_sel].reset_index(drop=True)
            cur, prof_row = tl.iloc[-1], tfs.iloc[-1]

            st.markdown(
                f"**{sid_sel}** · {prof_row['faculty'] or '-'} · {prof_row['program'] or '-'} "
                f"({prof_row['program_type']}) · รับเข้า: {prof_row['admission_method']} · "
                f"เข้าปี {'-' if pd.isna(prof_row['entry_year']) else int(prof_row['entry_year'])} · "
                f"ม.ปลาย {prof_row['hs_gpax']:.2f} · ทุน/กยศ.: "
                f"{'มี' if prof_row['scholarship'] else 'ไม่มี'}")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("ระดับล่าสุด", cur["ระดับ"], cur["ภาค"], delta_color="off")
            c2.metric("GPAX ล่าสุด", f"{cur['gpax']:.2f}")
            c3.metric("หน่วยกิตสะสม (CP)", f"{cur['cp_cum']} / {int(prof_row['required_credits'])}",
                      f"ช้ากว่าแผน {cur['credits_behind']}" if cur["credits_behind"] else "ตามแผน",
                      delta_color="inverse" if cur["credits_behind"] else "off")
            c4.metric("เตือนครั้งแรก", first_alert(timeline, sid_sel) or "ยังไม่เคย")
            known = is_known_at_risk(status_of.get(sid_sel, "ไม่ทราบ"))
            if known is not None:
                pred = cur["ระดับ"] in ("เสี่ยง", "เสี่ยงสูง")
                text = (f"ระดับล่าสุด '{cur['ระดับ']}' เทียบสถานภาพจริง "
                        f"'{status_of[sid_sel]}'")
                if pred == known:
                    st.success(text + " — ตรงกัน")
                else:
                    st.warning(text + " — ไม่ตรง ดูเหตุผลรายภาคประกอบ")

            # ---- กราฟ
            order = tl["ภาค"].tolist()
            gl = tl.melt(id_vars=["ภาค"], value_vars=["gpa", "gpax"], var_name="ชนิด",
                         value_name="ค่า").replace({"ชนิด": {"gpa": "GPA", "gpax": "GPAX"}})
            g_chart = alt.layer(
                alt.Chart(gl).mark_line(point=True).encode(
                    x=alt.X("ภาค:N", sort=order, title=None),
                    y=alt.Y("ค่า:Q", scale=alt.Scale(domain=[0, 4]), title=None),
                    color=alt.Color("ชนิด:N", legend=alt.Legend(title=None, orient="bottom")),
                    tooltip=["ภาค", "ชนิด", alt.Tooltip("ค่า:Q", format=".2f")]),
                alt.Chart(pd.DataFrame({"y": [PROBATION_GPAX]})).mark_rule(
                    color=RISE_COLOR, strokeDash=[4, 4]).encode(y="y:Q"),
            ).properties(height=240, width="container", title="GPA / GPAX (เส้นประ = 2.00)")
            plan = tfs["required_credits"] / tfs["plan_terms"] * tfs["term_no"]
            cdf = pd.concat([
                pd.DataFrame({"ภาค": order, "หน่วยกิต": tfs["cp_cum"], "ชนิด": "สะสมจริง (CP)"}),
                pd.DataFrame({"ภาค": order, "หน่วยกิต": plan.round(0), "ชนิด": "ตามแผน"})])
            c_chart = alt.Chart(cdf).mark_line(point=True).encode(
                x=alt.X("ภาค:N", sort=order, title=None),
                y=alt.Y("หน่วยกิต:Q", title=None),
                color=alt.Color("ชนิด:N", scale=alt.Scale(
                    domain=["สะสมจริง (CP)", "ตามแผน"], range=[FALL_COLOR, MUTED]),
                    legend=alt.Legend(title=None, orient="bottom", symbolType="stroke",
                                      symbolStrokeWidth=2)),
                strokeDash=alt.StrokeDash("ชนิด:N", scale=alt.Scale(
                    domain=["สะสมจริง (CP)", "ตามแผน"], range=[[1, 0], [4, 4]]), legend=None),
                tooltip=["ภาค", "ชนิด", "หน่วยกิต"],
            ).properties(height=240, width="container", title="หน่วยกิตสะสม เทียบแผน")
            r_chart = term_risk_chart(tl, "ความเสี่ยงจากโมเดล (สี = ระดับรวม)", None, 240)
            c1, c2, c3 = st.columns(3)
            c1.altair_chart(g_chart)
            c2.altair_chart(c_chart)
            c3.altair_chart(r_chart)

            # ---- ตารางแบบหน้าผลการเรียน
            # ภาคแรกไม่มีค่าโมเดล (ต้องใช้ 2 ภาคติดกัน) — เขียน "ใช้กฎ" แทนช่องว่าง/None
            model_txt = pd.to_numeric(tl["model_prob"], errors="coerce").map(
                lambda p: "ใช้กฎ" if pd.isna(p) else f"{p:.2f}")
            st.dataframe(
                tl[["ภาค", "cr", "cp", "gpa", "gpax", "n_f", "n_w", "term_result"]]
                .assign(model_prob=model_txt.values, ระดับ=tl["ระดับ"].values,
                        เหตุผล=tl["เหตุผล"].values)
                .rename(columns={
                    "cr": "CR", "cp": "CP", "gpa": "GPA", "gpax": "GPAX", "n_f": "F", "n_w": "W",
                    "term_result": "ผลการศึกษา (ทะเบียน)", "model_prob": "โมเดล"}),
                hide_index=True,
                column_config={"GPA": st.column_config.NumberColumn(format="%.2f"),
                               "GPAX": st.column_config.NumberColumn(format="%.2f"),
                               "เหตุผล": st.column_config.TextColumn(width="large")})

            # ---- เหตุผลของโมเดลในภาคล่าสุด (ภาษา มข.)
            st.markdown(f"**ปัจจัยที่โมเดลให้น้ำหนักในภาคล่าสุด ({cur['ภาค']})**")
            if source == "kku":
                kexpl = load_kku_explainer(kku_bundle)
                row = tfs.iloc[[-1]]
                facs = kexpl.top_factors(row[KKU_FEATURES], k=8)
                fdf = pd.DataFrame([{
                    "feature": f, "ปัจจัย": KKU_FEATURE_LABELS.get(f, f),
                    "ค่า": str(row[f].iloc[0]), "value": v} for f, v in facs])
                st.altair_chart(kku_factor_chart(fdf, kexpl.units))
                tips = collect_tips(fdf, tfs.iloc[-1], KKU_CONCERN, KKU_INTERVENTIONS)
            else:
                wrow = windows[(windows[ID_COL] == sid_sel) & (windows["term_no"] == cur["term_no"])]
                if wrow.empty:
                    st.caption("ต้องมีอย่างน้อย 2 ภาคจึงจะใช้โมเดลได้")
                    tips = []
                else:
                    uexpl = load_explainer(model, tuple(features))
                    sv = uexpl.shap_values(wrow[features]).iloc[0]
                    fdf, filled = factors_in_kku_terms(sv, tfs.iloc[-2], tfs.iloc[-1])
                    st.altair_chart(kku_factor_chart(fdf.head(8), uexpl.units))
                    st.caption(
                        f"แสดงเฉพาะปัจจัยที่มาจากข้อมูล มข. จริง · ฟีเจอร์ของชุด UCI ที่ มข. ไม่มี "
                        f"(คะแนนสอบเข้า, ตัวเลขเศรษฐกิจโปรตุเกส, รหัสสาขา ฯลฯ) ถูกเติมค่าคงที่ "
                        f"และรวมกันมีผล {filled:+.3f} — ไม่นำมาตีความ")
                    tips = collect_tips(fdf, wrow[features].iloc[0], wording=UCI_TIPS_IN_KKU)
            show_tips(tips, cur["ระดับ"], "ข้อเสนอแนะ")

            c1, c2 = st.columns(2)
            latest_uci = (windows.sort_values("term_no").groupby(ID_COL, sort=False).tail(1)
                          .drop(columns="term_no") if len(windows) else pd.DataFrame())
            if c1.button("เพิ่มเข้ารายชื่อในแดชบอร์ด", type="primary",
                         disabled=latest_uci.empty,
                         help="เพิ่มชั่วคราวในรายชื่อตัวอย่าง UCI (หายเมื่อปิดเว็บ) · ถ้าต้องการให้ขึ้นในรายชื่อหลักแบบถาวร ให้บันทึกลงฐานข้อมูล"):
                current = st.session_state.get("kku_rows")
                merged = latest_uci if current is None else pd.concat([current, latest_uci])
                st.session_state["kku_rows"] = merged.drop_duplicates(ID_COL, keep="last")
                st.session_state.pop("kku_preview", None)
                st.rerun()
            if c2.button("ล้างข้อมูลที่กรอก"):
                st.session_state.pop("kku_preview", None)
                st.rerun()

            if not is_admin:
                st.caption("การบันทึกลงฐานข้อมูลทำได้ในมุมมองผู้ดูแลระบบ")
            else:
                with st.expander("บันทึกลงฐานข้อมูล"):
                    consent = st.checkbox(
                        "ยืนยันว่าเป็นข้อมูลของตนเอง หรือได้รับความยินยอม/อนุญาตจากเจ้าของข้อมูลแล้ว "
                        "และไม่มีชื่อหรือรหัสนักศึกษาจริง", key="kku_consent")
                    src_label = st.selectbox("ที่มาของข้อมูล",
                                             ["กรอกเอง", "แบบฟอร์มยินยอม", "ระบบทะเบียน (ได้รับอนุญาต)"])
                    if st.button("บันทึก", disabled=not consent):
                        try:
                            n = db.save_students(preview, source=src_label, consent=consent)
                            load_db_students.clear()
                            db.save_assessments(timeline, source,
                                                float(kku_bundle["threshold"]) if kku_bundle
                                                else threshold)
                            st.success(f"บันทึกแล้ว {n} คน พร้อมผลประเมินรายภาค")
                        except Exception as e:
                            st.error(f"บันทึกไม่สำเร็จ: {e}")

    if kku_rows is not None and len(kku_rows):
        st.info(f"มีนักศึกษา มข. ในรายชื่อแดชบอร์ด {len(kku_rows)} คน: "
                f"{', '.join(kku_rows[ID_COL].astype(str))}")
        if st.button("ลบนักศึกษา มข. ออกจากรายชื่อทั้งหมด"):
            st.session_state.pop("kku_rows", None)
            st.rerun()

    st.caption(
        f"ข้อจำกัด: เกณฑ์ GPAX {PROBATION_GPAX:.2f}/{DISMISS_GPAX:.2f} และหน่วยกิตตามแผน "
        "ต้องตรวจกับระเบียบ มข. ของรุ่นนั้น · ถ้ายังไม่มีโมเดล มข. ชั้นโมเดลเป็นการยืมรูปแบบจาก UCI "
        "ผลนี้ใช้สาธิต ยังไม่ใช่การยืนยันความแม่นยำ จนกว่าจะทดสอบกับข้อมูล มข. ที่รู้ผลจริง")

render_footer()
