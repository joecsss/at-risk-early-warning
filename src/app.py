"""
app.py — แดชบอร์ดสำหรับอาจารย์ที่ปรึกษา (Streamlit)

รัน:  streamlit run src/app.py

หน้าที่ของระบบ: "สนับสนุนการตัดสินใจ" ของอาจารย์ — จัดลำดับนักศึกษาที่ควรเข้าไปดูแลก่อน
พร้อมเหตุผลประกอบ (SHAP) ไม่ใช่การตัดสินผลการเรียนอัตโนมัติ
"""
from __future__ import annotations

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
from feature_labels import GENERIC_INTERVENTION, decode_value, label_th, suggest
from kku_mapping import PARENT_EDU, PARENT_OCC, STATUS_OPTIONS
from kku_schema import CHOICES, TERM_FIELDS, dictionary_df
from kku_schema import COLUMNS as KKU_SCHEMA_COLUMNS
from kku_schema import template_df as kku_template_df
from kku_schema import validate as validate_kku_terms
from kku_timeline import (DISMISS_GPAX, KKU_FEATURE_LABELS, KKU_FEATURES, KKU_INTERVENTIONS,
                          LEVEL_COLORS, PROBATION_GPAX, build_timeline, factors_in_kku_terms,
                          first_alert, is_known_at_risk)

st.set_page_config(page_title="ระบบเตือนภัยล่วงหน้านักศึกษาเสี่ยง",
                   page_icon="🎓", layout="wide")

BAND_COLORS = {"เสี่ยงสูง": "#D64545", "เสี่ยงปานกลาง": "#E8A33D",
               "เฝ้าระวัง": "#4C78A8", "ปกติ": "#7A8B99"}


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
                range=["#D64545", "#2E9E6B"]), legend=alt.Legend(title=None)),
            tooltip=["ปัจจัย", "ค่าของนักศึกษา",
                     alt.Tooltip("value:Q", title="ค่า SHAP", format="+.3f")],
        ).properties(height=320, width="container",
                     title="ปัจจัยที่มีผลต่อความเสี่ยงมากที่สุด (SHAP)")
    )
    return chart, factors


def render_suggestions(factors, row: pd.DataFrame) -> None:
    st.subheader("💡 ข้อเสนอแนะการช่วยเหลือ")
    pushers = [(f, v) for f, v in factors if v > 0][:4]
    if not pushers:
        st.success("ไม่พบปัจจัยที่ผลักดันความเสี่ยงอย่างมีนัยสำคัญสำหรับนักศึกษาคนนี้")
    for f, v in pushers:
        st.markdown(
            f"- **{label_th(f)}** (ค่าปัจจุบัน: {decode_value(f, row[f].iloc[0])}, "
            f"อิทธิพล {v:+.2f}) → {suggest(f)}")
    st.caption("ข้อเสนอแนะสร้างจากปัจจัยที่โมเดลให้น้ำหนัก — "
               "โปรดใช้ประกอบดุลยพินิจและข้อมูลจริงของนักศึกษาเสมอ")


# ------------------------------------------------------------------- เริ่มโหลด
bundle = load_model()
model, features = bundle["model"], bundle["features"]
default_threshold = float(bundle.get("threshold", 0.5))
metrics = load_metrics()

st.title("🎓 ระบบเตือนภัยล่วงหน้าสำหรับนักศึกษาที่เสี่ยงด้านผลการเรียน")
st.caption("เครื่องมือสนับสนุนการตัดสินใจของอาจารย์ที่ปรึกษา — "
           "ผลการทำนายเป็นเพียงสัญญาณเตือน ไม่ใช่ข้อสรุปเกี่ยวกับตัวนักศึกษา")

# ------------------------------------------------------------------- แถบข้างซ้าย
with st.sidebar:
    st.info("**เวอร์ชันต้นแบบ** · รายชื่อนักศึกษาเป็นชุดทดสอบสาธารณะ UCI (รหัสสมมติ) · "
            "ข้อมูลที่กรอกในแท็บ มข. อยู่ในหน่วยความจำของเซสชันเท่านั้น ไม่ถูกบันทึก")
    st.header("⚙️ การตั้งค่า")

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
        value=round(default_threshold, 2), step=0.01,
        help="ลดค่านี้ = จับกลุ่มเสี่ยงได้ครบขึ้น (recall สูง) แต่แจ้งเตือนเกินจริงมากขึ้น",
    )

    st.divider()
    st.subheader("ℹ️ ข้อมูลโมเดล")
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
n_total = len(work)
n_risk = int((work["risk"] >= threshold).sum())
c1, c2, c3, c4 = st.columns(4)
c1.metric("นักศึกษาทั้งหมด", f"{n_total:,} คน")
c2.metric("เข้าเกณฑ์เสี่ยง", f"{n_risk:,} คน", f"{n_risk / max(n_total, 1) * 100:.1f}%")
c3.metric("เสี่ยงสูง (ควรพบด่วน)", f"{int((work['band'] == 'เสี่ยงสูง').sum()):,} คน")
if TARGET_COL in work.columns:
    y = work[TARGET_COL]
    pred = (work["risk"] >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    rec = tp / max(int((y == 1).sum()), 1)
    prec = tp / max(int((pred == 1).sum()), 1)
    c4.metric("Recall / Precision ที่เกณฑ์นี้", f"{rec:.2f} / {prec:.2f}",
              help="คำนวณจากข้อมูลชุดนี้ที่มีเฉลยจริง")
else:
    c4.metric("ค่าความเสี่ยงเฉลี่ย", f"{work['risk'].mean() * 100:.1f}%")

tab_list, tab_person, tab_model, tab_kku = st.tabs(
    ["📋 รายชื่อกลุ่มเสี่ยง", "🔍 อธิบายรายบุคคล", "📊 ภาพรวมโมเดล", "🏫 ข้อมูลจริง มข."])

# ------------------------------------------------------------ แท็บ 1: รายชื่อ
with tab_list:
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
        "⬇️ ดาวน์โหลดรายชื่อ (CSV)",
        table.to_csv(index=False).encode("utf-8-sig"),
        file_name="at_risk_students.csv", mime="text/csv",
    )

# -------------------------------------------------------- แท็บ 2: รายบุคคล
with tab_person:
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
        st.caption(f"เกณฑ์ปัจจุบัน: {threshold:.2f} — "
                   f"{'เข้าเกณฑ์ต้องติดตาม' if p >= threshold else 'ยังไม่เข้าเกณฑ์'}")
        for c in ("Course", "Age at enrollment", "Tuition fees up to date",
                  "Debtor", "Scholarship holder"):
            if c in row.columns:
                st.write(f"**{label_th(c)}:** {decode_value(c, row[c].iloc[0])}")

    with right:
        expl = load_explainer(model, tuple(features))
        chart, factors = factor_chart(expl, row)
        st.altair_chart(chart)

    render_suggestions(factors, row)

# ------------------------------------------------------- แท็บ 3: ภาพรวมโมเดล
with tab_model:
    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.subheader("ปัจจัยสำคัญระดับภาพรวม (mean |SHAP|)")
        expl = load_explainer(model, tuple(features))
        imp = global_importance(expl, df, tuple(features))
        idf = pd.DataFrame({"ปัจจัย": [label_th(i) for i in imp.index],
                            "ความสำคัญ": imp.values})
        st.altair_chart(
            alt.Chart(idf).mark_bar(color="#4C78A8").encode(
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

    if TARGET_COL in work.columns:
        st.subheader(f"ตารางความสับสน (confusion matrix) ที่เกณฑ์ {threshold:.2f}")
        y = work[TARGET_COL]
        pred = (work["risk"] >= threshold).astype(int)
        cm = pd.DataFrame(
            [[int(((pred == 0) & (y == 0)).sum()), int(((pred == 1) & (y == 0)).sum())],
             [int(((pred == 0) & (y == 1)).sum()), int(((pred == 1) & (y == 1)).sum())]],
            index=["จริง: ไม่เสี่ยง", "จริง: เสี่ยง"],
            columns=["ทาย: ไม่เสี่ยง", "ทาย: เสี่ยง"])
        st.dataframe(cm)
        st.caption("ช่อง 'จริง: เสี่ยง / ทาย: ไม่เสี่ยง' คือนักศึกษาที่ระบบพลาด — "
                   "ลดเกณฑ์เพื่อให้พลาดน้อยลง แลกกับการแจ้งเตือนเกินจริงที่มากขึ้น")

# --------------------------------------------------- แท็บ 4: ข้อมูลจริง มข.
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
            domain=["เพิ่มความเสี่ยง", "ลดความเสี่ยง"], range=["#D64545", "#2E9E6B"]),
            legend=alt.Legend(title=None)),
        tooltip=["ปัจจัย", "ค่า", alt.Tooltip("value:Q", title="ค่า SHAP", format="+.3f")],
    ).properties(height=300, width="container")


def yes_no(v) -> int:
    return int(v == "ใช่")


with tab_kku:
    kku_bundle = load_kku_bundle()
    st.subheader("🏫 ข้อมูลจริงตามระบบทะเบียน มข. — ประเมินทุกภาค")
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

    with st.expander("📖 ข้อมูล มข. ที่ระบบใช้ (พจนานุกรมข้อมูล)"):
        ddf = dictionary_df()
        st.dataframe(ddf, hide_index=True)
        st.caption(f"ทั้งหมด {len(ddf)} ฟิลด์ · เข้าโมเดล UCI ได้ {int((ddf['โมเดล UCI'] == '✓').sum())} · "
                   f"ใช้ในกฎเตือนภัย {int((ddf['กฎเตือนภัย'] == '✓').sum())} · "
                   f"ใช้ในโมเดล มข. {int((ddf['โมเดล มข.'] == '✓').sum())}")

    mode = st.radio("วิธีนำเข้า", ["กรอกทีละคน", "อัปโหลดไฟล์ CSV รูปแบบทะเบียน"],
                    horizontal=True)

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
        st.download_button(
            "⬇️ ไฟล์ตัวอย่าง (1 แถว = 1 คน × 1 ภาค)",
            kku_template_df().to_csv(index=False).encode("utf-8-sig"),
            file_name="kku_template_terms.csv", mime="text/csv")
        st.caption("คอลัมน์บังคับดูได้จากพจนานุกรมข้อมูลด้านบน คอลัมน์อื่นเว้นว่างได้ · "
                   "ข้อมูลโปรไฟล์ใส่แค่แถวแรกของแต่ละคนก็ได้ · ค่าที่เป็นตัวเลือก: "
                   + " | ".join(f"{k}: {' / '.join(o for o in v if o)}"
                                for k, v in CHOICES.items()
                                if k in ("term_result", "admission_method", "family_income")))
        up = st.file_uploader("อัปโหลดไฟล์ CSV", type="csv", key="kku_upload")
        if up is not None:
            st.session_state["kku_preview"] = pd.read_csv(up)


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
                    color="#D64545", strokeDash=[4, 4]).encode(y="y:Q"),
            ).properties(height=240, width="container", title="GPA / GPAX (เส้นประ = 2.00)")
            plan = tfs["required_credits"] / tfs["plan_terms"] * tfs["term_no"]
            cdf = pd.concat([
                pd.DataFrame({"ภาค": order, "หน่วยกิต": tfs["cp_cum"], "ชนิด": "สะสมจริง (CP)"}),
                pd.DataFrame({"ภาค": order, "หน่วยกิต": plan.round(0), "ชนิด": "ตามแผน"})])
            c_chart = alt.Chart(cdf).mark_line(point=True).encode(
                x=alt.X("ภาค:N", sort=order, title=None),
                y=alt.Y("หน่วยกิต:Q", title=None),
                color=alt.Color("ชนิด:N", scale=alt.Scale(range=["#2E9E6B", "#999"]),
                                legend=alt.Legend(title=None, orient="bottom")),
                strokeDash=alt.StrokeDash("ชนิด:N", legend=None),
                tooltip=["ภาค", "ชนิด", "หน่วยกิต"],
            ).properties(height=240, width="container", title="หน่วยกิตสะสม เทียบแผน")
            r_chart = alt.Chart(tl).mark_bar().encode(
                x=alt.X("ภาค:N", sort=order, title=None),
                y=alt.Y("model_prob:Q", scale=alt.Scale(domain=[0, 1]),
                        axis=alt.Axis(format="%"), title=None),
                color=alt.Color("ระดับ:N", scale=alt.Scale(
                    domain=list(LEVEL_COLORS), range=list(LEVEL_COLORS.values())),
                    legend=alt.Legend(title=None, orient="bottom")),
                tooltip=["ภาค", "ระดับ", alt.Tooltip("model_prob:Q", format=".0%"), "เหตุผล"],
            ).properties(height=240, width="container",
                         title="ความเสี่ยงจากโมเดล (สี = ระดับรวม)")
            c1, c2, c3 = st.columns(3)
            c1.altair_chart(g_chart)
            c2.altair_chart(c_chart)
            c3.altair_chart(r_chart)

            # ---- ตารางแบบหน้าผลการเรียน
            st.dataframe(
                tl[["ภาค", "cr", "cp", "gpa", "gpax", "n_f", "n_w", "term_result",
                    "model_prob", "ระดับ", "เหตุผล"]].rename(columns={
                    "cr": "CR", "cp": "CP", "gpa": "GPA", "gpax": "GPAX", "n_f": "F", "n_w": "W",
                    "term_result": "ผลการศึกษา (ทะเบียน)", "model_prob": "โมเดล"}),
                hide_index=True,
                column_config={"GPA": st.column_config.NumberColumn(format="%.2f"),
                               "GPAX": st.column_config.NumberColumn(format="%.2f"),
                               "โมเดล": st.column_config.NumberColumn(format="%.2f"),
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
                tips = [(r.ปัจจัย, r.ค่า, KKU_INTERVENTIONS.get(r.feature, GENERIC_INTERVENTION))
                        for r in fdf.itertuples() if r.value > 0][:4]
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
                    tips = [(r.ปัจจัย, r.ค่า, suggest(r.feature))
                            for r in fdf.itertuples() if r.value > 0][:4]
            if tips:
                st.markdown("**💡 ข้อเสนอแนะ**")
                for name, val, tip in tips:
                    st.markdown(f"- **{name}** (ค่า: {val}) → {tip}")

            c1, c2 = st.columns(2)
            latest_uci = (windows.sort_values("term_no").groupby(ID_COL, sort=False).tail(1)
                          .drop(columns="term_no") if len(windows) else pd.DataFrame())
            if c1.button("➕ เพิ่มเข้ารายชื่อในแดชบอร์ด", type="primary",
                         disabled=latest_uci.empty,
                         help="รายชื่อหลักใช้โมเดล UCI กับ 2 ภาคล่าสุด"):
                current = st.session_state.get("kku_rows")
                merged = latest_uci if current is None else pd.concat([current, latest_uci])
                st.session_state["kku_rows"] = merged.drop_duplicates(ID_COL, keep="last")
                st.session_state.pop("kku_preview", None)
                st.rerun()
            if c2.button("ล้างข้อมูลที่กรอก"):
                st.session_state.pop("kku_preview", None)
                st.rerun()

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

st.divider()
st.caption("⚠️ ข้อพึงระวังเชิงจริยธรรม: ผลทำนายอาจมีอคติจากข้อมูลในอดีต "
           "ควรใช้เพื่อ 'จัดลำดับการเข้าไปช่วยเหลือ' เท่านั้น "
           "ไม่ควรใช้ตัดสิทธิ์หรือเปิดเผยต่อบุคคลที่ไม่เกี่ยวข้อง")
