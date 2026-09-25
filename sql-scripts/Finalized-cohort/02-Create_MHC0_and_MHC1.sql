-- Finalized cohort construction: stage 2
--
-- Prerequisite:
--   Run 01-Load_and_create_base_cohort.sql first.
--
-- Definitions used here:
--   MHC0 is a patient-level "psychiatric-code negative" cohort. A patient is
--   eligible only if none of their MIMIC-IV hospital admissions contains a
--   psychiatric ICD code at any diagnosis position.
--
--   MHC1 is an admission-level "documented mental-health context" cohort. An
--   eligible physical admission is included when either:
--     * an earlier hospital admission contains a psychiatric ICD code at any
--       diagnosis position; or
--     * the current physical admission contains a secondary psychiatric ICD
--       code (seq_num > 1).
--
-- These definitions make MHC0 and MHC1 mutually exclusive at both the subject
-- and admission level. Patients who satisfy neither definition remain outside
-- both cohorts.


-- ---------------------------------------------------------------------------
-- 1. MHC0: no psychiatric ICD code anywhere in the observed trajectory
-- ---------------------------------------------------------------------------

CREATE OR REPLACE TABLE finalized_MHC0 AS
SELECT
    b.subject_id,
    b.hadm_id,
    b.admittime
FROM finalized_base_physical_admissions b
WHERE NOT EXISTS (
    SELECT 1
    FROM diagnoses_icd d_any
    JOIN psychiatric_icd_codes p_any
        ON d_any.icd_version = p_any.icd_version
       AND d_any.icd_code = p_any.icd_code
    WHERE d_any.subject_id = b.subject_id
);


-- ---------------------------------------------------------------------------
-- 2. MHC1: prior psychiatric history and/or a current secondary psychiatric ICD
-- ---------------------------------------------------------------------------

-- No seq_num restriction is placed on the earlier diagnosis. Therefore, both
-- primary and non-primary psychiatric diagnoses establish prior history.
--
-- The current admission remains an eligible physical admission, so its primary
-- diagnosis is nonpsychiatric. A current psychiatric diagnosis can therefore
-- qualify only from a non-primary position (seq_num > 1).
CREATE OR REPLACE TABLE finalized_MHC1 AS
WITH base_with_history_flags AS (
    SELECT
        b.subject_id,
        b.hadm_id,
        b.admittime,

        CASE
            WHEN EXISTS (
                SELECT 1
                FROM diagnoses_icd d_current
                JOIN psychiatric_icd_codes p_current
                    ON d_current.icd_version = p_current.icd_version
                   AND d_current.icd_code = p_current.icd_code
                WHERE d_current.subject_id = b.subject_id
                  AND d_current.hadm_id = b.hadm_id
                  AND d_current.seq_num > 1
            ) THEN 1
            ELSE 0
        END AS has_current_secondary_psychiatric_icd,

        CASE
            WHEN EXISTS (
                SELECT 1
                FROM admissions a_previous
                JOIN diagnoses_icd d_previous
                    ON a_previous.subject_id = d_previous.subject_id
                   AND a_previous.hadm_id = d_previous.hadm_id
                JOIN psychiatric_icd_codes p_previous
                    ON d_previous.icd_version = p_previous.icd_version
                   AND d_previous.icd_code = p_previous.icd_code
                WHERE a_previous.subject_id = b.subject_id
                  AND a_previous.hadm_id <> b.hadm_id
                  AND a_previous.admittime < b.admittime
            ) THEN 1
            ELSE 0
        END AS has_prior_psychiatric_history

    FROM finalized_base_physical_admissions b
)

SELECT
    subject_id,
    hadm_id,
    admittime,
    has_prior_psychiatric_history,
    has_current_secondary_psychiatric_icd,
    CASE
        WHEN has_prior_psychiatric_history = 1
         AND has_current_secondary_psychiatric_icd = 0
            THEN 'prior_history_only'
        WHEN has_prior_psychiatric_history = 1
         AND has_current_secondary_psychiatric_icd = 1
            THEN 'prior_history_and_current_secondary'
        WHEN has_prior_psychiatric_history = 0
         AND has_current_secondary_psychiatric_icd = 1
            THEN 'current_secondary_only'
    END AS mhc1_version
FROM base_with_history_flags
WHERE has_prior_psychiatric_history = 1
   OR has_current_secondary_psychiatric_icd = 1;


-- ---------------------------------------------------------------------------
-- 3. Cohort sizes
-- ---------------------------------------------------------------------------

SELECT
    'MHC0' AS cohort,
    COUNT(DISTINCT subject_id) AS n_subjects,
    COUNT(DISTINCT hadm_id) AS n_admissions
FROM finalized_MHC0

UNION ALL

SELECT
    'MHC1' AS cohort,
    COUNT(DISTINCT subject_id) AS n_subjects,
    COUNT(DISTINCT hadm_id) AS n_admissions
FROM finalized_MHC1

ORDER BY cohort;


-- ---------------------------------------------------------------------------
-- 4. Counts for the three MHC1 versions
-- ---------------------------------------------------------------------------

-- These versions are mutually exclusive at the admission level. A patient may
-- appear in more than one version across different physical admissions, so the
-- distinct-subject counts should not be added together.
SELECT
    mhc1_version,
    COUNT(DISTINCT subject_id) AS n_subjects,
    COUNT(DISTINCT hadm_id) AS n_admissions
FROM finalized_MHC1
GROUP BY mhc1_version
ORDER BY mhc1_version;


-- ---------------------------------------------------------------------------
-- 5. Mutual-exclusivity checks
-- ---------------------------------------------------------------------------

-- Expected result: zero overlapping subjects and zero overlapping admissions.
SELECT
    COUNT(DISTINCT m0.subject_id) AS n_overlapping_subjects,
    COUNT(DISTINCT CASE
        WHEN m0.hadm_id = m1.hadm_id THEN m0.hadm_id
    END) AS n_overlapping_admissions
FROM finalized_MHC0 m0
JOIN finalized_MHC1 m1
    ON m0.subject_id = m1.subject_id;


-- ---------------------------------------------------------------------------
-- 6. Disposition of every patient in the prefiltered physical-admission base
-- ---------------------------------------------------------------------------

-- A patient is "represented in MHC1" when at least one eligible physical
-- admission contains a current secondary psychiatric diagnosis or occurs after
-- a psychiatric-coded admission. Earlier physical admissions without either
-- form of psychiatric context are not MHC1 rows.
--
-- The excluded group contains patients who have at least one psychiatric ICD
-- code somewhere in MIMIC-IV, but no eligible physical admission with either
-- prior psychiatric history or a current secondary psychiatric diagnosis.
CREATE OR REPLACE TABLE finalized_cohort_subject_disposition AS
WITH base_subjects AS (
    SELECT
        subject_id,
        COUNT(DISTINCT hadm_id) AS n_base_admissions
    FROM finalized_base_physical_admissions
    GROUP BY subject_id
),

mhc0_subjects AS (
    SELECT
        subject_id,
        COUNT(DISTINCT hadm_id) AS n_mhc0_admissions
    FROM finalized_MHC0
    GROUP BY subject_id
),

mhc1_subjects AS (
    SELECT
        subject_id,
        COUNT(DISTINCT hadm_id) AS n_mhc1_admissions
    FROM finalized_MHC1
    GROUP BY subject_id
)

SELECT
    b.subject_id,
    b.n_base_admissions,
    COALESCE(m0.n_mhc0_admissions, 0) AS n_mhc0_admissions,
    COALESCE(m1.n_mhc1_admissions, 0) AS n_mhc1_admissions,
    b.n_base_admissions
        - COALESCE(m0.n_mhc0_admissions, 0)
        - COALESCE(m1.n_mhc1_admissions, 0)
        AS n_unassigned_base_admissions,
    CASE
        WHEN m0.subject_id IS NOT NULL THEN 'MHC0'
        WHEN m1.subject_id IS NOT NULL THEN 'MHC1'
        ELSE 'excluded_psych_code_but_no_context_positive_base_admission'
    END AS subject_disposition
FROM base_subjects b
LEFT JOIN mhc0_subjects m0
    ON b.subject_id = m0.subject_id
LEFT JOIN mhc1_subjects m1
    ON b.subject_id = m1.subject_id;


-- Patient counts for the three mutually exclusive dispositions.
SELECT
    subject_disposition,
    COUNT(*) AS n_subjects,
    SUM(n_base_admissions) AS n_base_admissions,
    SUM(n_mhc0_admissions) AS n_mhc0_admissions,
    SUM(n_mhc1_admissions) AS n_mhc1_admissions,
    SUM(n_unassigned_base_admissions) AS n_unassigned_base_admissions
FROM finalized_cohort_subject_disposition
GROUP BY subject_disposition
ORDER BY subject_disposition;
