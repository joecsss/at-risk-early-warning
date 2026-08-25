"""
explain.py — อธิบายผลของแบบจำลองด้วย SHAP (ภาพรวม + รายบุคคล)

จุดสำคัญ:
  * SHAP ต้องมี "ข้อมูลอ้างอิง (background)" ที่เป็นตัวแทนประชากร ถ้าใช้แถวเดียวเป็น
    background ค่า SHAP จะออกมาเป็น 0 ทั้งหมด (เทียบกับตัวเอง) — โค้ดนี้สุ่มจากชุด train
  * เลือก explainer ให้ตรงชนิดโมเดล (Linear/Tree) เพื่อความเร็วและความแม่นยำ
  * ถ้าโมเดลมี one-hot encoding จะรวมค่า SHAP ของแต่ละหมวดกลับเป็น "ฟีเจอร์ต้นฉบับ"
    เพื่อให้อาจารย์อ่านผลได้ในหน่วยที่เข้าใจง่าย

รัน:  python src/explain.py     # พิมพ์ตัวอย่าง + บันทึกกราฟลง reports/
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

import joblib
from common import MODEL_FILE, REPORT_DIR, TEST_CSV, TRAIN_CSV, setup_console

DEFAULT_BACKGROUND = 200
LINEAR_MODELS = ("LogisticRegression", "LinearSVC", "RidgeClassifier", "SGDClassifier")
TREE_MODELS = (
    "RandomForestClassifier", "GradientBoostingClassifier", "ExtraTreesClassifier",
    "HistGradientBoostingClassifier", "DecisionTreeClassifier",
)


def load_bundle() -> dict:
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"ไม่พบ {MODEL_FILE} — รัน `python src/train.py` ก่อน")
    return joblib.load(MODEL_FILE)


def _steps_and_estimator(model):
    """แยกขั้นตอนแปลงข้อมูล (ที่มี .transform) ออกจากตัวจำแนกขั้นสุดท้าย."""
    if not isinstance(model, Pipeline):
        return [], model
    transformers = [s for _, s in model.steps[:-1] if hasattr(s, "transform")]
    return transformers, model.steps[-1][1]


def _source_feature(name: str, features: list[str]) -> str:
    """แปลงชื่อคอลัมน์หลังแปลง (เช่น 'Course_9500') กลับเป็นฟีเจอร์ต้นฉบับ ('Course')."""
    if name in features:
        return name
    candidates = [c for c in features if name.startswith(c + "_")]
    return max(candidates, key=len) if candidates else name


def _positive_class(values: np.ndarray) -> np.ndarray:
    """ดึงค่า SHAP ของคลาส 'เสี่ยง' (บางโมเดลคืนค่ามาทั้งสองคลาส)."""
    values = np.asarray(values)
    if values.ndim == 3:          # (n, features, classes)
        return values[:, :, -1]
    return values


class RiskExplainer:
    """ห่อ SHAP explainer ให้ใช้งานง่ายทั้งในสคริปต์และในแดชบอร์ด."""

    def __init__(self, model, features: list[str], background: pd.DataFrame,
                 max_background: int = DEFAULT_BACKGROUND, seed: int = 42):
        self.model = model
        self.features = list(features)
        self.transformers, self.estimator = _steps_and_estimator(model)

        bg = background[self.features]
        if len(bg) > max_background:
            bg = bg.sample(max_background, random_state=seed)
        self.background_raw = bg
        bg_t = self.transform(bg)
        self.out_names = list(bg_t.columns)
        self.sources = [_source_feature(n, self.features) for n in self.out_names]

        kind = type(self.estimator).__name__
        if kind in TREE_MODELS:
            self.explainer = shap.TreeExplainer(self.estimator)
            self.units = "log-odds"
        elif kind in LINEAR_MODELS:
            masker = shap.maskers.Independent(bg_t, max_samples=len(bg_t))
            self.explainer = shap.LinearExplainer(self.estimator, masker)
            self.units = "log-odds"
        else:  # เผื่อโมเดลชนิดอื่น — ช้ากว่าแต่ใช้ได้ทั่วไป
            masker = shap.maskers.Independent(bg_t, max_samples=100)
            self.explainer = shap.Explainer(self.estimator.predict_proba, masker)
            self.units = "probability"

    # ------------------------------------------------------------------ core
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """ผ่านขั้นตอน preprocessing ของ pipeline แล้วคืน DataFrame ที่มีชื่อคอลัมน์."""
        out = X[self.features]
        for step in self.transformers:
            names = None
            if hasattr(step, "get_feature_names_out"):
                try:
                    names = list(step.get_feature_names_out(list(out.columns)))
                except Exception:
                    names = None
            arr = step.transform(out)
            arr = arr.toarray() if hasattr(arr, "toarray") else np.asarray(arr)
            out = pd.DataFrame(
                arr, columns=names or [f"f{i}" for i in range(arr.shape[1])],
                index=X.index,
            )
        return out

    def shap_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """ค่า SHAP รวมกลับเป็นฟีเจอร์ต้นฉบับ (แถว = นักศึกษา, คอลัมน์ = ฟีเจอร์)."""
        Xt = self.transform(X)
        raw = self.explainer(Xt) if not isinstance(self.explainer, shap.TreeExplainer) \
            else self.explainer(Xt, check_additivity=False)
        vals = _positive_class(raw.values)
        wide = pd.DataFrame(vals, columns=self.out_names, index=X.index)
        return wide.T.groupby(self.sources).sum().T[self._ordered_sources()]

    def _ordered_sources(self) -> list[str]:
        seen, order = set(), []
        for s in self.sources:
            if s not in seen:
                seen.add(s)
                order.append(s)
        return order

    # ------------------------------------------------------------- ระดับรายคน
    def top_factors(self, X_row: pd.DataFrame, k: int = 5) -> list[tuple[str, float]]:
        """k ปัจจัยที่มีอิทธิพลสูงสุดต่อความเสี่ยงของนักศึกษา 1 คน (เรียงตามขนาด)."""
        sv = self.shap_values(X_row).iloc[0]
        order = sv.abs().sort_values(ascending=False).index[:k]
        return [(f, float(sv[f])) for f in order]

    # -------------------------------------------------------------- ระดับภาพรวม
    def global_importance(self, X: pd.DataFrame, k: int = 20) -> pd.Series:
        """ความสำคัญเฉลี่ย |SHAP| ของแต่ละฟีเจอร์ในภาพรวม."""
        return self.shap_values(X).abs().mean().sort_values(ascending=False).head(k)


def save_global_plots(explainer: RiskExplainer, X: pd.DataFrame,
                      n_sample: int = 400, seed: int = 42) -> list[str]:
    """บันทึกกราฟภาพรวมสำหรับใส่ในรายงาน (ชื่อฟีเจอร์เป็นภาษาอังกฤษตามข้อมูลต้นฉบับ)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(REPORT_DIR, exist_ok=True)
    sample = X.sample(min(n_sample, len(X)), random_state=seed)
    sv = explainer.shap_values(sample)
    paths = []

    imp = sv.abs().mean().sort_values(ascending=False).head(20)[::-1]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(imp.index, imp.values, color="#4C78A8")
    ax.set_xlabel(f"mean |SHAP value| ({explainer.units})")
    ax.set_title("Global feature importance (SHAP)")
    fig.tight_layout()
    p = os.path.join(REPORT_DIR, "shap_global_bar.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(p)

    fig = plt.figure()
    shap.summary_plot(sv.values, sample[sv.columns], max_display=20, show=False)
    plt.title("SHAP summary (impact on dropout risk)")
    plt.tight_layout()
    p = os.path.join(REPORT_DIR, "shap_summary.png")
    plt.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(p)
    return paths


def build_default_explainer() -> tuple[RiskExplainer, pd.DataFrame, dict]:
    bundle = load_bundle()
    features = bundle["features"]
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    expl = RiskExplainer(bundle["model"], features, train)
    return expl, test, bundle


if __name__ == "__main__":
    setup_console()
    expl, test, bundle = build_default_explainer()
    X = test[bundle["features"]]
    print(f"โมเดล: {bundle['model_name']} | feature_set: {bundle['feature_set']} | "
          f"threshold: {bundle['threshold']:.3f}")

    proba = bundle["model"].predict_proba(X)[:, 1]
    idx = int(np.argmax(proba))
    print(f"\nตัวอย่าง: นักศึกษาที่เสี่ยงสูงสุดในชุดทดสอบ "
          f"(ความเสี่ยง {proba[idx] * 100:.1f}%)")
    for name, val in expl.top_factors(X.iloc[[idx]]):
        arrow = "เพิ่มความเสี่ยง" if val > 0 else "ลดความเสี่ยง"
        print(f"  {name:<45}{val:+.3f}  {arrow}")

    print("\nปัจจัยสำคัญระดับภาพรวม (mean |SHAP|):")
    for name, val in expl.global_importance(X, k=10).items():
        print(f"  {name:<45}{val:.3f}")

    for p in save_global_plots(expl, X):
        print(f"บันทึกกราฟ: {p}")
