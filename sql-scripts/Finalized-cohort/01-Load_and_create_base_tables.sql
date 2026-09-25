-- Finalized cohort construction: stage 1
--
-- This script:
--   1. loads the required MIMIC-IV hosp and note tables;
--   2. reports the initial number of subjects and admissions;
--   3. defines the general psychiatric and grey-zone ICD lookup tables;
--   4. creates the eligible physical-admission base table; and
--   5. reports the number of subjects and admissions after prefiltering.
--
-- IMPORTANT:
-- Psychiatric and grey-zone exclusions are applied to the PRIMARY diagnosis.
-- A secondary psychiatric ICD code does not remove an otherwise eligible
-- physical admission at this stage. It can be used in a later cohort step.

-- ---------------------------------------------------------------------------
-- 1. Load the required source tables
-- ---------------------------------------------------------------------------

@set mimic_dirs=/Users/licseng/Downloads/thesis/physionet.org/files

CREATE TABLE IF NOT EXISTS patients AS
SELECT *
FROM read_csv_auto('${mimic_dirs}/mimiciv/3.1/hosp/patients.csv.gz');

CREATE TABLE IF NOT EXISTS admissions AS
SELECT *
FROM read_csv_auto('${mimic_dirs}/mimiciv/3.1/hosp/admissions.csv.gz');

CREATE TABLE IF NOT EXISTS diagnoses_icd AS
SELECT *
FROM read_csv_auto('${mimic_dirs}/mimiciv/3.1/hosp/diagnoses_icd.csv.gz');

CREATE TABLE IF NOT EXISTS d_icd_diagnoses AS
SELECT *
FROM read_csv_auto('${mimic_dirs}/mimiciv/3.1/hosp/d_icd_diagnoses.csv.gz');

CREATE TABLE IF NOT EXISTS discharge AS
SELECT *
FROM read_csv_auto('${mimic_dirs}/mimic-iv-note/2.2/note/discharge.csv.gz');


-- ---------------------------------------------------------------------------
-- 2. Initial MIMIC-IV hosp population
-- ---------------------------------------------------------------------------

SELECT
    COUNT(DISTINCT subject_id) AS n_subjects_initial,
    COUNT(DISTINCT hadm_id) AS n_admissions_initial
FROM admissions;


-- ---------------------------------------------------------------------------
-- 3. ICD lookup tables required for the prefilter
-- ---------------------------------------------------------------------------

-- General psychiatric ICD codes, including suicide/self-harm diagnoses.
CREATE OR REPLACE TABLE psychiatric_icd_codes AS
SELECT DISTINCT
    icd_version,
    icd_code,
    long_title
FROM d_icd_diagnoses
WHERE
    (
        icd_version = 10
        AND icd_code LIKE 'F%'
    )
    OR
    (
        icd_version = 9
        AND (
            regexp_matches(icd_code, '^(29[0-9]|30[0-9]|31[0-9])')
            OR icd_code LIKE 'V11%'
        )
    )
    OR lower(long_title) LIKE '%suicid%'
    OR lower(long_title) LIKE '%self-harm%'
    OR lower(long_title) LIKE '%self harm%'
    OR lower(long_title) LIKE '%intentional self%';

-- Ambiguous physical diagnoses excluded when they are the primary diagnosis.
CREATE OR REPLACE TABLE grey_zone_physical_icd_codes AS
SELECT DISTINCT
    icd_version,
    icd_code,
    long_title
FROM d_icd_diagnoses
WHERE
    (
        icd_version = 9
        AND icd_code IN (
            '3575', '4255', '53530', '53531',
            '5710', '5711', '5712', '5713'
        )
    )
    OR
    (
        icd_version = 10
        AND icd_code IN (
            'G621', 'G721', 'I426', 'K292', 'K2920', 'K2921',
            'K70', 'K700', 'K701', 'K7010', 'K7011', 'K702',
            'K703', 'K7030', 'K7031', 'K704', 'K7040', 'K7041',
            'K709', 'K860'
        )
    )

UNION

SELECT DISTINCT
    d.icd_version,
    d.icd_code,
    d.long_title
FROM d_icd_diagnoses d
LEFT JOIN psychiatric_icd_codes p
    ON d.icd_version = p.icd_version
   AND d.icd_code = p.icd_code
WHERE p.icd_code IS NULL
  AND (
        lower(d.long_title) LIKE '%poisoning%'
        OR lower(d.long_title) LIKE '%overdose%'
        OR lower(d.long_title) LIKE '%toxic effect%'
        OR lower(d.long_title) LIKE '%drug-induced%'
        OR lower(d.long_title) LIKE '%medication-induced%'
        OR lower(d.long_title) LIKE '%withdrawal%'
        OR lower(d.long_title) LIKE '%intoxication%'
  );


-- ---------------------------------------------------------------------------
-- 4. Prefiltered physical-admission base table
-- ---------------------------------------------------------------------------

-- Retained admissions must:
--   * have a matching discharge note;
--   * have a primary ICD diagnosis (seq_num = 1);
--   * not have a psychiatric primary ICD diagnosis; and
--   * not have a grey-zone physical primary ICD diagnosis.
CREATE OR REPLACE TABLE finalized_base_physical_admissions AS
SELECT DISTINCT
    a.subject_id,
    a.hadm_id,
    a.admittime
FROM admissions a
JOIN discharge di
    ON a.subject_id = di.subject_id
   AND a.hadm_id = di.hadm_id
JOIN diagnoses_icd d1
    ON a.subject_id = d1.subject_id
   AND a.hadm_id = d1.hadm_id
   AND d1.seq_num = 1
WHERE NOT EXISTS (
    SELECT 1
    FROM psychiatric_icd_codes p
    WHERE p.icd_version = d1.icd_version
      AND p.icd_code = d1.icd_code
)
AND NOT EXISTS (
    SELECT 1
    FROM grey_zone_physical_icd_codes g
    WHERE g.icd_version = d1.icd_version
      AND g.icd_code = d1.icd_code
);


-- ---------------------------------------------------------------------------
-- 5. Population after prefiltering
-- ---------------------------------------------------------------------------

SELECT
    COUNT(DISTINCT subject_id) AS n_subjects_after_prefilter,
    COUNT(DISTINCT hadm_id) AS n_admissions_after_prefilter
FROM finalized_base_physical_admissions;
