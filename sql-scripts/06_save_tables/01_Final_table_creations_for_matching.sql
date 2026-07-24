-- without any mental health context and history control group
CREATE OR REPLACE TABLE export_only_MHC0 AS
WITH mhc0_with_covariates AS (
    SELECT
        b.subject_id,
        b.hadm_id,
        b.admittime,

        pat.gender AS sex,
        pat.anchor_age,
        pat.anchor_year,
        pat.anchor_age
            + (EXTRACT(YEAR FROM b.admittime) - pat.anchor_year)
            AS age_at_admission,

        b.MHC,
        b.has_secondary_psychiatric_same_admission,
        b.has_prior_psychiatric_history,

        d.note_id,
        d.charttime,
        d.storetime,
        d.text

    FROM base_admissions_mhc_subject_groups b

    JOIN patients pat
        ON b.subject_id = pat.subject_id

    JOIN discharge d
        ON b.subject_id = d.subject_id
       AND b.hadm_id = d.hadm_id

    WHERE b.subject_group = 'only_MHC0'
)

SELECT *
FROM mhc0_with_covariates
WHERE sex IS NOT NULL
  AND sex IN ('F', 'M')
  AND age_at_admission IS NOT NULL
  AND age_at_admission BETWEEN 18 AND 120;

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
),

mhh_with_covariates AS (
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
       AND p.hadm_id = d.hadm_id
)

SELECT *
FROM mhh_with_covariates
WHERE sex IS NOT NULL
  AND sex IN ('F', 'M')
  AND age_at_admission IS NOT NULL
  AND age_at_admission BETWEEN 18 AND 120;

--MHC0 ICD code lists
CREATE OR REPLACE TABLE admission_icd_lists_only_MHC0 AS
SELECT
    d.subject_id,
    d.hadm_id,
    d.icd_version,
    d.icd_code
FROM diagnoses_icd d
JOIN (
    SELECT DISTINCT subject_id, hadm_id
    FROM export_only_MHC0
) mhc0
    ON d.subject_id = mhc0.subject_id
   AND d.hadm_id = mhc0.hadm_id;

--MHH1-psychotic ICD code lists
CREATE OR REPLACE TABLE admission_icd_lists_MHH_psychotic AS
SELECT
    d.subject_id,
    d.hadm_id,
    d.icd_version,
    d.icd_code
FROM diagnoses_icd d
JOIN (
    SELECT DISTINCT subject_id, hadm_id
    FROM export_MHH_psychotic
) mhh
    ON d.subject_id = mhh.subject_id
   AND d.hadm_id = mhh.hadm_id;


