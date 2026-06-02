--MHC1-same-admission case group (HISTORY-ONLY IS EXCLUDED!)
CREATE OR REPLACE TABLE export_only_MHC1_same_admission AS
SELECT
b.subject_id,
    b.hadm_id,
    b.admittime,
    b.MHC,
    b.has_secondary_psychiatric_same_admission,
    b.has_prior_psychiatric_history,
    d.note_id,
    d.charttime,
    d.storetime,
    d.text
FROM base_admissions_mhc_subject_groups b
JOIN discharge d
    ON b.subject_id = d.subject_id
   AND b.hadm_id = d.hadm_id
WHERE b.subject_group = 'only_MHC1'
  AND b.has_secondary_psychiatric_same_admission = 1;

--without any mental health context and history control group
CREATE OR REPLACE TABLE export_only_MHC0 AS
SELECT
    b.subject_id,
    b.hadm_id,
    b.admittime,
    b.MHC,
    b.has_secondary_psychiatric_same_admission,
    b.has_prior_psychiatric_history,
    d.note_id,
    d.charttime,
    d.storetime,
    d.text
FROM base_admissions_mhc_subject_groups b
JOIN discharge d
    ON b.subject_id = d.subject_id
   AND b.hadm_id = d.hadm_id
WHERE b.subject_group = 'only_MHC0';

--Mixed group for trajectory and sentiment analysis
CREATE OR REPLACE TABLE export_mixed_group AS
SELECT
    b.subject_id,
    b.hadm_id,
    b.admittime,
    b.MHC,
    b.has_secondary_psychiatric_same_admission,
    b.has_prior_psychiatric_history,
    d.note_id,
    d.charttime,
    d.storetime,
    d.text
FROM base_admissions_mhc_subject_groups b
JOIN discharge d
    ON b.subject_id = d.subject_id
   AND b.hadm_id = d.hadm_id
WHERE b.subject_group = 'MHC_0_to_1';