@set mimic_dirs=

CREATE TABLE IF NOT EXISTS patients AS
SELECT * FROM read_csv_auto('${mimic_dirs}/mimiciv/3.1/hosp/patients.csv.gz');

CREATE TABLE IF NOT EXISTS admissions AS
SELECT * FROM read_csv_auto('${mimic_dirs}/mimiciv/3.1/hosp/admissions.csv.gz');

CREATE TABLE IF NOT EXISTS diagnoses_icd AS
SELECT * FROM read_csv_auto('${mimic_dirs}/mimiciv/3.1/hosp/diagnoses_icd.csv.gz');

CREATE TABLE IF NOT EXISTS d_icd_diagnoses AS
SELECT * FROM read_csv_auto('${mimic_dirs}/mimiciv/3.1/hosp/d_icd_diagnoses.csv.gz');



CREATE TABLE IF NOT EXISTS discharge AS
SELECT * FROM read_csv_auto('${mimic_dirs}/mimic-iv-note/2.2/note/discharge.csv.gz');
