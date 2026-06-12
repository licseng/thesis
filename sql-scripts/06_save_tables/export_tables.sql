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

--MHH = 1 group
CREATE OR REPLACE TABLE export_MHH_focus4 AS
WITH focus4_flags AS (
    SELECT
        p.subject_id,
        p.hadm_id,
        CASE WHEN lower(p.prior_psych_categories) LIKE '%internalizing%' THEN 1 ELSE 0 END AS has_internalizing_history,
        CASE WHEN lower(p.prior_psych_categories) LIKE '%substance_related%' THEN 1 ELSE 0 END AS has_substance_related_history,
        CASE WHEN lower(p.prior_psych_categories) LIKE '%psychotic%' THEN 1 ELSE 0 END AS has_psychotic_history,
        CASE WHEN lower(p.prior_psych_categories) LIKE '%personality_behavioral%' THEN 1 ELSE 0 END AS has_personality_behavioral_history
    FROM possible_overshadowing_admissions_psychiatric p
    WHERE p.has_prior_psychiatric_history = 1
      AND (
          lower(p.prior_psych_categories) LIKE '%internalizing%'
          OR lower(p.prior_psych_categories) LIKE '%substance_related%'
          OR lower(p.prior_psych_categories) LIKE '%psychotic%'
          OR lower(p.prior_psych_categories) LIKE '%personality_behavioral%'
      )
)
SELECT
    p.subject_id,
    p.hadm_id,
    p.admittime,
    p.has_secondary_psychiatric_same_admission,
    p.has_prior_psychiatric_history,
    p.same_admission_psych_categories,
    p.prior_psych_categories,

    f.has_internalizing_history,
    f.has_substance_related_history,
    f.has_psychotic_history,
    f.has_personality_behavioral_history,

    d.note_id,
    d.charttime,
    d.storetime,
    d.text
FROM possible_overshadowing_admissions_psychiatric p
JOIN focus4_flags f
    ON p.subject_id = f.subject_id
   AND p.hadm_id = f.hadm_id
JOIN discharge d
    ON p.subject_id = d.subject_id
   AND p.hadm_id = d.hadm_id;
    
    
SELECT
    COUNT(DISTINCT hadm_id) AS n_admissions,
    COUNT(DISTINCT subject_id) AS n_subjects
FROM export_MHH_focus4;

-- MHH = 1 group with psychotic history only
CREATE OR REPLACE TABLE export_MHH_psychotic AS
WITH psychotic_history_flags AS (
    SELECT DISTINCT
        p.subject_id,
        p.hadm_id,
        1 AS has_psychotic_history
    FROM possible_overshadowing_admissions_psychiatric p
    WHERE p.has_prior_psychiatric_history = 1
      AND lower(p.prior_psych_categories) LIKE '%psychotic%'
)
SELECT
    p.subject_id,
    p.hadm_id,
    p.admittime,

    pat.gender AS sex,
    pat.anchor_age,
    pat.anchor_year,
    pat.anchor_age
        + (EXTRACT(YEAR FROM p.admittime) - pat.anchor_year)
        AS age_at_admission,

    p.has_secondary_psychiatric_same_admission,
    p.has_prior_psychiatric_history,
    p.same_admission_psych_categories,
    p.prior_psych_categories,

    f.has_psychotic_history,

    d.note_id,
    d.charttime,
    d.storetime,
    d.text

FROM possible_overshadowing_admissions_psychiatric p

JOIN psychotic_history_flags f
    ON p.subject_id = f.subject_id
   AND p.hadm_id = f.hadm_id

JOIN patients pat
    ON p.subject_id = pat.subject_id

JOIN discharge d
    ON p.subject_id = d.subject_id
   AND p.hadm_id = d.hadm_id;

-- History-only cohort: any prior psychiatric history, no same-admission psychiatric ICD code
CREATE OR REPLACE TABLE export_MHH_history_only AS
SELECT
    p.subject_id,
    p.hadm_id,
    p.admittime,
    p.has_secondary_psychiatric_same_admission,
    p.has_prior_psychiatric_history,
    p.same_admission_psych_categories,
    p.prior_psych_categories,
    d.note_id,
    d.charttime,
    d.storetime,
    d.text
FROM possible_overshadowing_admissions_psychiatric p
JOIN discharge d
    ON p.subject_id = d.subject_id
   AND p.hadm_id = d.hadm_id
WHERE p.has_prior_psychiatric_history = 1
  AND p.has_secondary_psychiatric_same_admission = 0;

SELECT
    COUNT(DISTINCT hadm_id) AS n_admissions,
    COUNT(DISTINCT subject_id) AS n_subjects
FROM export_MHH_history_only;

