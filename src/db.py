"""
db.py — ฐานข้อมูลของระบบตามแผนภาพ ER ในเล่ม: STUDENT, TERM_RECORD, RISK_ASSESSMENT

รองรับ 2 แบบ ด้วยโค้ดและ schema เดียวกัน:
  * SQLite ในเครื่อง (ค่าเริ่มต้น) — ไฟล์ data/kku_app.db (ถูกกันออกจาก git)
  * PostgreSQL ออนไลน์ (เช่น Supabase/Neon) — เมื่อตั้งค่า DATABASE_URL
    ใน .streamlit/secrets.toml, ใน Secrets ของ Streamlit Cloud หรือเป็น environment variable

บรรทัดคำสั่ง (ใช้ฐานข้อมูลตามที่ตั้งค่าไว้ ถ้าไม่ได้ตั้งจะใช้ SQLite):
    python src/db.py --stats                       # มีข้อมูลกี่คน กี่ภาค ใช้ฐานข้อมูลไหน
    python src/db.py --import data/kku_xxx.csv --source registrar --consent   # รับได้ทั้ง 2 แบบ
    python src/db.py --export data/backup.csv      # ส่งออก CSV แบบ 1 คน 1 แถว (--per-term = 1 แถวต่อเทอม)
    python src/db.py --delete KKU-001              # ลบ 1 คน (พร้อมผลการเรียนและผลการประเมิน)
    python src/db.py --copy-from-sqlite            # คัดลอกข้อมูลจาก SQLite ในเครื่องขึ้นฐานข้อมูลออนไลน์

เก็บได้เฉพาะข้อมูลที่ได้รับอนุญาตหรือความยินยอมแล้ว · ข้อมูลจริงของนักศึกษาห้ามขึ้นฐานข้อมูลออนไลน์
จนกว่ามหาวิทยาลัยจะอนุญาตให้ประมวลผลนอกระบบของมหาวิทยาลัย
"""
from __future__ import annotations

import argparse
import math
import os
import sqlite3
import sys
import warnings
from contextlib import contextmanager
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

from common import DATA_DIR, ROOT, setup_console
from kku_schema import COLUMNS, PROFILE_FIELDS, TERM_FIELDS, normalize, validate

DB_FILE = os.environ.get("KKU_DB_FILE", os.path.join(DATA_DIR, "kku_app.db"))
_URL: str | None = os.environ.get("DATABASE_URL") or None
_READY: set[str] = set()          # ฐานข้อมูลที่สร้างตารางแล้วในโปรเซสนี้

PROFILE_COLS = [c for c in PROFILE_FIELDS if c != "student_id"]
TERM_COLS = [c for c in TERM_FIELDS if c != "term_no"]

_NUMERIC = {"entry_year", "required_credits", "plan_terms", "hs_gpax", "age_at_enrollment",
            "male", "single", "same_province", "disability", "health_issue",
            "siblings_studying", "scholarship", "n_courses", "n_f", "n_w", "cr", "cp", "gpa",
            "gpax", "outstanding_debt", "late_fine", "logins"}


# ------------------------------------------------------------ เลือกฐานข้อมูล
def configure(url: str | None) -> None:
    """ตั้ง DATABASE_URL (postgresql://...) — ส่ง None หรือ "" เพื่อใช้ SQLite ในเครื่อง."""
    global _URL
    _URL = (url or "").strip() or None


def is_online() -> bool:
    return _URL is not None


def describe() -> str:
    if _URL:
        host = _URL.split("@")[-1].split("/")[0]           # ไม่แสดงชื่อผู้ใช้/รหัสผ่าน
        return f"PostgreSQL ออนไลน์ ({host})"
    return f"SQLite ในเครื่อง ({os.path.relpath(DB_FILE, ROOT)})"


def _schema(pg: bool) -> list[str]:
    num = "DOUBLE PRECISION" if pg else "REAL"
    typ = lambda c: num if c in _NUMERIC else "TEXT"            # noqa: E731
    auto = "BIGSERIAL PRIMARY KEY" if pg else "INTEGER PRIMARY KEY AUTOINCREMENT"
    return [
        f"""CREATE TABLE IF NOT EXISTS student (
            student_id TEXT PRIMARY KEY,
            {", ".join(f"{c} {typ(c)}" for c in PROFILE_COLS)},
            source TEXT,
            consent INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL)""",
        f"""CREATE TABLE IF NOT EXISTS term_record (
            student_id TEXT NOT NULL REFERENCES student(student_id) ON DELETE CASCADE,
            term_no INTEGER NOT NULL,
            {", ".join(f"{c} {typ(c)}" for c in TERM_COLS)},
            PRIMARY KEY (student_id, term_no))""",
        f"""CREATE TABLE IF NOT EXISTS risk_assessment (
            assessment_id {auto},
            student_id TEXT NOT NULL REFERENCES student(student_id) ON DELETE CASCADE,
            term_no INTEGER NOT NULL,
            model_source TEXT,
            risk_score {num},
            risk_level TEXT,
            reasons TEXT,
            threshold {num},
            assessed_at TEXT NOT NULL)""",
        "CREATE INDEX IF NOT EXISTS idx_risk_student ON risk_assessment(student_id, term_no)",
    ] + ([
        # Supabase เปิดตารางใน schema public ผ่าน REST API อัตโนมัติ — เปิด Row Level Security
        # (ไม่มี policy) เพื่อปิดทางนั้น แอปเชื่อมต่อในฐานะเจ้าของตารางจึงยังใช้งานได้ตามปกติ
        "ALTER TABLE student ENABLE ROW LEVEL SECURITY",
        "ALTER TABLE term_record ENABLE ROW LEVEL SECURITY",
        "ALTER TABLE risk_assessment ENABLE ROW LEVEL SECURITY",
    ] if pg else [])


class _Con:
    """ห่อ connection ให้ใช้ SQL แบบเดียวกัน (ตัวแทนค่า ?) ได้ทั้ง SQLite และ PostgreSQL."""

    def __init__(self, raw, pg: bool):
        self.raw, self.pg = raw, pg

    def _q(self, sql: str) -> str:
        return sql.replace("?", "%s") if self.pg else sql

    def execute(self, sql: str, params=()):
        cur = self.raw.cursor()
        cur.execute(self._q(sql), tuple(params))
        return cur

    def executemany(self, sql: str, rows):
        rows = [tuple(r) for r in rows]
        if rows:
            self.raw.cursor().executemany(self._q(sql), rows)

    def read(self, sql: str, params=()) -> pd.DataFrame:
        with warnings.catch_warnings():       # pandas เตือนเมื่อใช้ connection ของ psycopg2 ตรง ๆ
            warnings.simplefilter("ignore", UserWarning)
            return pd.read_sql_query(self._q(sql), self.raw, params=list(params) or None)


@contextmanager
def connect(path: str | None = None):
    """เปิดฐานข้อมูล (สร้างตารางครั้งแรก) แล้ว commit และปิดให้เมื่อจบ block
    path ใช้กับ SQLite เท่านั้น (เช่นตอนทดสอบหรือคัดลอกข้อมูล)."""
    pg = _URL is not None and path is None
    if pg:
        try:
            import psycopg2
        except ImportError as e:
            raise RuntimeError("ต้องติดตั้ง psycopg2-binary ก่อนใช้ฐานข้อมูลออนไลน์ "
                               "(pip install psycopg2-binary)") from e
        raw = psycopg2.connect(_URL, connect_timeout=10)
        key = _URL
    else:
        path = path or DB_FILE
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        raw = sqlite3.connect(path)
        raw.execute("PRAGMA foreign_keys = ON")
        key = os.path.abspath(path)
    con = _Con(raw, pg)
    try:
        if key not in _READY:
            for stmt in _schema(pg):
                con.execute(stmt)
            raw.commit()
            _READY.add(key)
        yield con
        raw.commit()
    except Exception:
        raw.rollback()
        raise
    finally:
        raw.close()


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _clean(v):
    """แปลงค่าจาก pandas/numpy ให้ฐานข้อมูลเก็บได้ (NaN -> NULL)."""
    if v is None:
        return None
    if hasattr(v, "item"):          # numpy scalar
        v = v.item()
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


# --------------------------------------------------------------- เขียนข้อมูล
def save_students(raw: pd.DataFrame, source: str = "dashboard", consent: bool = False,
                  path: str | None = None) -> int:
    """บันทึกข้อมูลรูปแบบทะเบียน มข. (ตรวจก่อน) แทนที่ผลการเรียนเดิมของคนเดียวกันทั้งชุด
    ประวัติผลการประเมินเดิมยังอยู่ · คืนจำนวนนักศึกษาที่บันทึก."""
    errors = validate(raw)
    if errors:
        raise ValueError("ข้อมูลไม่ถูกต้อง: " + "; ".join(errors[:5]))
    d = normalize(raw)
    now = _now()
    cols = PROFILE_COLS + ["source", "consent"]
    upsert = (f"INSERT INTO student (student_id, {', '.join(cols)}, created_at, updated_at) "
              f"VALUES ({', '.join('?' * (len(cols) + 3))}) "
              f"ON CONFLICT (student_id) DO UPDATE SET "
              f"{', '.join(f'{c} = EXCLUDED.{c}' for c in cols)}, updated_at = EXCLUDED.updated_at")
    ins_term = (f"INSERT INTO term_record (student_id, term_no, {', '.join(TERM_COLS)}) "
                f"VALUES ({', '.join('?' * (len(TERM_COLS) + 2))})")
    with connect(path) as con:
        for sid, g in d.groupby("student_id", sort=False):
            prof = g.iloc[0]
            con.execute(upsert, [sid, *[_clean(prof[c]) for c in PROFILE_COLS],
                                 source, int(consent), now, now])
            con.execute("DELETE FROM term_record WHERE student_id = ?", (sid,))
            con.executemany(ins_term, [[sid, int(r["term_no"]),
                                        *[_clean(r[c]) for c in TERM_COLS]]
                                       for _, r in g.iterrows()])
    return int(d["student_id"].nunique())


def save_assessments(timeline: pd.DataFrame, model_source: str, threshold: float,
                     path: str | None = None) -> int:
    """บันทึกผลประเมินรายภาค (ผลจาก kku_timeline.build_timeline) เป็นประวัติ."""
    now = _now()
    rows = [[str(r["student_id"]), int(r["term_no"]), model_source,
             _clean(float(r["model_prob"])) if pd.notna(r["model_prob"]) else None,
             r["ระดับ"], r["เหตุผล"], float(threshold), now]
            for _, r in timeline.iterrows()]
    with connect(path) as con:
        con.executemany(
            "INSERT INTO risk_assessment (student_id, term_no, model_source, risk_score, "
            "risk_level, reasons, threshold, assessed_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
    return len(rows)


def delete_student(student_id: str, path: str | None = None) -> None:
    with connect(path) as con:
        con.execute("DELETE FROM student WHERE student_id = ?", (student_id,))


def delete_all(path: str | None = None) -> None:
    with connect(path) as con:
        con.execute("DELETE FROM student")


# ---------------------------------------------------------------- อ่านข้อมูล
def load_students(student_ids: list[str] | None = None, path: str | None = None) -> pd.DataFrame:
    """อ่านกลับเป็นรูปแบบทะเบียน มข. (1 แถว = 1 คน × 1 ภาค) พร้อมใช้กับ kku_timeline/train_kku."""
    q = (f"SELECT t.student_id, t.term_no, {', '.join('t.' + c for c in TERM_COLS)}, "
         f"{', '.join('s.' + c for c in PROFILE_COLS)} "
         "FROM term_record t JOIN student s ON s.student_id = t.student_id")
    params: list = []
    if student_ids:
        q += f" WHERE t.student_id IN ({', '.join('?' * len(student_ids))})"
        params = list(student_ids)
    q += " ORDER BY t.student_id, t.term_no"
    with connect(path) as con:
        df = con.read(q, params)
    return df.reindex(columns=COLUMNS)


def list_students(path: str | None = None) -> pd.DataFrame:
    """สรุปรายคน: จำนวนภาค, ภาคล่าสุด, ระดับล่าสุดที่ประเมินไว้, วันที่แก้ไข."""
    q = """
    SELECT s.student_id, s.status, s.source, s.consent, s.updated_at,
           COUNT(t.term_no) AS n_terms, MAX(t.term_no) AS last_term,
           (SELECT r.risk_level FROM risk_assessment r
             WHERE r.student_id = s.student_id
             ORDER BY r.assessed_at DESC, r.term_no DESC LIMIT 1) AS last_level
    FROM student s LEFT JOIN term_record t ON t.student_id = s.student_id
    GROUP BY s.student_id, s.status, s.source, s.consent, s.updated_at
    ORDER BY s.updated_at DESC
    """
    with connect(path) as con:
        return con.read(q)


def assessment_history(student_id: str | None = None, path: str | None = None) -> pd.DataFrame:
    q = ("SELECT student_id, term_no, risk_level, risk_score, model_source, threshold, "
         "reasons, assessed_at FROM risk_assessment")
    params: list = []
    if student_id:
        q += " WHERE student_id = ?"
        params = [student_id]
    q += " ORDER BY assessed_at, student_id, term_no"
    with connect(path) as con:
        return con.read(q, params)


def stats(path: str | None = None) -> dict:
    with connect(path) as con:
        n_s = con.execute("SELECT COUNT(*) FROM student").fetchone()[0]
        n_t = con.execute("SELECT COUNT(*) FROM term_record").fetchone()[0]
        n_a = con.execute("SELECT COUNT(*) FROM risk_assessment").fetchone()[0]
    where = path if path else describe()
    return {"students": n_s, "term_records": n_t, "assessments": n_a, "database": where}


def copy_from_sqlite(sqlite_path: str | None = None) -> tuple[int, int]:
    """คัดลอกนักศึกษา ผลการเรียน และประวัติการประเมินจาก SQLite ในเครื่องไปฐานข้อมูลที่ตั้งค่าไว้."""
    if not is_online():
        raise RuntimeError("ยังไม่ได้ตั้ง DATABASE_URL — ไม่มีฐานข้อมูลปลายทาง")
    src = sqlite_path or DB_FILE
    data = load_students(path=src)
    if data.empty:
        return 0, 0
    with connect(src) as con:
        meta = con.read("SELECT student_id, source, consent FROM student")
        hist = con.read("SELECT student_id, term_no, model_source, risk_score, risk_level, "
                        "reasons, threshold, assessed_at FROM risk_assessment")
    n = 0
    for (source, consent), ids in meta.groupby(["source", "consent"])["student_id"]:
        part = data[data["student_id"].isin(ids)]
        if len(part):
            n += save_students(part, source=source or "sqlite", consent=bool(consent))
    with connect() as con:
        con.executemany(
            "INSERT INTO risk_assessment (student_id, term_no, model_source, risk_score, "
            "risk_level, reasons, threshold, assessed_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [[_clean(v) for v in r] for r in hist.itertuples(index=False)])
    return n, len(hist)


# ------------------------------------------------------------- บรรทัดคำสั่ง
def _url_from_secrets() -> str | None:
    """ให้บรรทัดคำสั่งอ่าน DATABASE_URL จาก .streamlit/secrets.toml ได้เหมือนแอป."""
    path = os.path.join(ROOT, ".streamlit", "secrets.toml")
    if not os.path.exists(path):
        return None
    try:
        import tomllib                        # Python 3.11+
    except ImportError:
        return None
    with open(path, "rb") as f:
        return (tomllib.load(f).get("DATABASE_URL") or "").strip() or None


def main() -> None:
    setup_console()
    ap = argparse.ArgumentParser(description="จัดการฐานข้อมูลระบบเตือนภัย")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--import", dest="import_csv")
    ap.add_argument("--export", dest="export_csv")
    ap.add_argument("--delete")
    ap.add_argument("--copy-from-sqlite", action="store_true",
                    help="คัดลอกข้อมูลจาก data/kku_app.db ขึ้นฐานข้อมูลออนไลน์")
    ap.add_argument("--local", action="store_true", help="บังคับใช้ SQLite ในเครื่อง")
    ap.add_argument("--per-term", action="store_true",
                    help="ส่งออกแบบ 1 แถวต่อเทอม (ค่าเริ่มต้นคือ 1 คน 1 แถว)")
    ap.add_argument("--source", default="import", help="ที่มาของข้อมูล เช่น registrar, consent-form")
    ap.add_argument("--consent", action="store_true", help="ยืนยันว่าได้รับอนุญาต/ยินยอมแล้ว")
    a = ap.parse_args()
    if not a.local and not is_online():
        configure(_url_from_secrets())
    if a.local:
        configure(None)
    print(f"ฐานข้อมูล: {describe()}")
    if a.copy_from_sqlite:
        n, h = copy_from_sqlite()
        print(f"คัดลอก {n} คน และประวัติการประเมิน {h} รายการ")
    if a.import_csv:
        n = save_students(pd.read_csv(a.import_csv), source=a.source, consent=a.consent)
        print(f"นำเข้า {n} คน")
    if a.export_csv:
        out = load_students()
        if not a.per_term:
            from kku_schema import to_wide
            out = to_wide(out)
        out.to_csv(a.export_csv, index=False, encoding="utf-8-sig")
        print(f"ส่งออก: {a.export_csv}")
    if a.delete:
        delete_student(a.delete)
        print(f"ลบ {a.delete}")
    if a.stats or not (a.import_csv or a.export_csv or a.delete or a.copy_from_sqlite):
        print(stats())


if __name__ == "__main__":
    main()
