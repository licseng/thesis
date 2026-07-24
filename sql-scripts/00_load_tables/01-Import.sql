@set mimic_dirs=/Users/licseng/Downloads/thesis/physionet.org/files

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




@set cohort_dirs=/Users/licseng/Downloads/thesis/thesis_code/python-code/

CREATE TABLE IF NOT EXISTS matched_cohort AS
SELECT * FROM read_csv_auto('${cohort_dirs}02_cohort_matching/matched_cohort_output/matched_admission_ids_for_dbeaver.csv');
