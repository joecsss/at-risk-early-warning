# ข้อมูล (Data)

ไฟล์ข้อมูลส่วนใหญ่ไม่ถูก commit เข้า git — โหลด/สร้างเองตามด้านล่าง

ยกเว้น `processed_train.csv` และ `processed_test.csv` ซึ่งสร้างจากชุดสาธารณะ UCI (สัญญาอนุญาต CC BY 4.0, Realinho et al.) และ commit ไว้เพื่อให้เดโมออนไลน์ทำงานได้ ห้าม commit ไฟล์ข้อมูลจริงของนักศึกษา มข. เด็ดขาด

## ค่าเริ่มต้น: UCI "Predict Students' Dropout and Academic Success"
`src/data_prep.py` จะดึงให้อัตโนมัติผ่านไลบรารี `ucimlrepo` (id = 697)
ถ้าต้องการโหลดเอง: https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success

## ทางเลือก: OULAD (Open University Learning Analytics Dataset)
เหมาะกับการทำนายเชิงเวลา (early-warning ตามสัปดาห์) — https://analyse.kmi.open.ac.uk/open_dataset

## ไฟล์ที่ถูกสร้าง (หลังรัน data_prep.py)
- `raw_uci697.csv` — แคชข้อมูลดิบ (ใช้ซ้ำโดยไม่ต้องต่ออินเทอร์เน็ต; ดึงใหม่ด้วย `--refresh`)
- `processed_train.csv`
- `processed_test.csv`
- `metadata.json` — รายชื่อฟีเจอร์ที่ใช้จริง, ชุดฟีเจอร์ (`all`/`early`), จำนวนตัวอย่าง

คอลัมน์เป้าหมาย: `at_risk` — 1 = เสี่ยง (Dropout), 0 = ไม่เสี่ยง
คอลัมน์ `student_id` เป็นรหัสจำลองสำหรับอ้างอิงในแดชบอร์ด (ไม่ถูกใช้เป็นฟีเจอร์)
