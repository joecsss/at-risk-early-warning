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
from common import ID_COL, METRICS_JSON, MODEL_FILE, TARGET_COL, TEST_CSV, TRAIN_CSV
from explain import RiskExplainer
from feature_labels import decode_value, label_th, suggest

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

tab_list, tab_person, tab_model = st.tabs(
    ["📋 รายชื่อกลุ่มเสี่ยง", "🔍 อธิบายรายบุคคล", "📊 ภาพรวมโมเดล"])

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
        factors = expl.top_factors(row[features], k=8)
        fdf = pd.DataFrame(factors, columns=["feature", "value"])
        fdf["ปัจจัย"] = fdf["feature"].map(label_th)
        fdf["ค่าของนักศึกษา"] = [
            str(decode_value(f, row[f].iloc[0])) for f in fdf["feature"]]
        fdf["ทิศทาง"] = np.where(fdf["value"] > 0, "เพิ่มความเสี่ยง", "ลดความเสี่ยง")

        chart = (
            alt.Chart(fdf)
            .mark_bar()
            .encode(
                x=alt.X("value:Q", title=f"อิทธิพลต่อความเสี่ยง ({expl.units})"),
                y=alt.Y("ปัจจัย:N", sort=None, title=None),
                color=alt.Color("ทิศทาง:N", scale=alt.Scale(
                    domain=["เพิ่มความเสี่ยง", "ลดความเสี่ยง"],
                    range=["#D64545", "#2E9E6B"]), legend=alt.Legend(title=None)),
                tooltip=["ปัจจัย", "ค่าของนักศึกษา",
                         alt.Tooltip("value:Q", title="ค่า SHAP", format="+.3f")],
            )
            .properties(height=320, width="container",
                        title="ปัจจัยที่มีผลต่อความเสี่ยงมากที่สุด (SHAP)")
        )
        st.altair_chart(chart)

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

st.divider()
st.caption("⚠️ ข้อพึงระวังเชิงจริยธรรม: ผลทำนายอาจมีอคติจากข้อมูลในอดีต "
           "ควรใช้เพื่อ 'จัดลำดับการเข้าไปช่วยเหลือ' เท่านั้น "
           "ไม่ควรใช้ตัดสิทธิ์หรือเปิดเผยต่อบุคคลที่ไม่เกี่ยวข้อง")
