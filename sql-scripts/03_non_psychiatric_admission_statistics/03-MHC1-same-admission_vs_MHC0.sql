-- Summarizes the final case-control cohorts and exports subject-level admission counts and ages for downstream distribution testing.
WITH case_cohort AS (
    SELECT
        b.subject_id,
        b.hadm_id
    FROM base_admissions_mhc_subject_groups b
    WHERE b.subject_group = 'only_MHC1'
      AND b.has_secondary_psychiatric_same_admission = 1
),
control_cohort AS (
    SELECT
        b.subject_id,
        b.hadm_id
    FROM base_admissions_mhc_subject_groups b
    WHERE b.subject_group = 'only_MHC0'
),
case_subject_admission_counts AS (
    SELECT
        subject_id,
        COUNT(DISTINCT hadm_id) AS n_admissions
    FROM case_cohort
    GROUP BY subject_id
),
control_subject_admission_counts AS (
    SELECT
        subject_id,
        COUNT(DISTINCT hadm_id) AS n_admissions
    FROM control_cohort
    GROUP BY subject_id
),
case_summary AS (
    SELECT
        'Case (MHC=1 same-admission only)' AS group_name,
        (SELECT COUNT(DISTINCT hadm_id) FROM case_cohort) AS n_admissions,
        COUNT(*) AS n_subjects,

        ROUND(AVG(p.anchor_age), 2) AS mean_anchor_age,
        ROUND(VAR_SAMP(p.anchor_age), 2) AS var_anchor_age,
        ROUND(STDDEV_SAMP(p.anchor_age), 2) AS sd_anchor_age,
        MIN(p.anchor_age) AS min_anchor_age,
        quantile_cont(p.anchor_age, 0.25) AS q1_anchor_age,
        quantile_cont(p.anchor_age, 0.5) AS median_anchor_age,
        quantile_cont(p.anchor_age, 0.75) AS q3_anchor_age,
        MAX(p.anchor_age) AS max_anchor_age,

        ROUND(AVG(c.n_admissions), 3) AS mean_admissions_per_subject,
        ROUND(VAR_SAMP(c.n_admissions), 3) AS var_admissions_per_subject,
        ROUND(STDDEV_SAMP(c.n_admissions), 3) AS sd_admissions_per_subject,
        MIN(c.n_admissions) AS min_admissions_per_subject,
        quantile_cont(c.n_admissions, 0.25) AS q1_admissions_per_subject,
        quantile_cont(c.n_admissions, 0.5) AS median_admissions_per_subject,
        quantile_cont(c.n_admissions, 0.75) AS q3_admissions_per_subject,
        MAX(c.n_admissions) AS max_admissions_per_subject
    FROM case_subject_admission_counts c
    JOIN patients p
        ON c.subject_id = p.subject_id
),
control_summary AS (
    SELECT
        'Control (MHC=0)' AS group_name,
        (SELECT COUNT(DISTINCT hadm_id) FROM control_cohort) AS n_admissions,
        COUNT(*) AS n_subjects,

        ROUND(AVG(p.anchor_age), 2) AS mean_anchor_age,
        ROUND(VAR_SAMP(p.anchor_age), 2) AS var_anchor_age,
        ROUND(STDDEV_SAMP(p.anchor_age), 2) AS sd_anchor_age,
        MIN(p.anchor_age) AS min_anchor_age,
        quantile_cont(p.anchor_age, 0.25) AS q1_anchor_age,
        quantile_cont(p.anchor_age, 0.5) AS median_anchor_age,
        quantile_cont(p.anchor_age, 0.75) AS q3_anchor_age,
        MAX(p.anchor_age) AS max_anchor_age,

        ROUND(AVG(c.n_admissions), 3) AS mean_admissions_per_subject,
        ROUND(VAR_SAMP(c.n_admissions), 3) AS var_admissions_per_subject,
        ROUND(STDDEV_SAMP(c.n_admissions), 3) AS sd_admissions_per_subject,
        MIN(c.n_admissions) AS min_admissions_per_subject,
        quantile_cont(c.n_admissions, 0.25) AS q1_admissions_per_subject,
        quantile_cont(c.n_admissions, 0.5) AS median_admissions_per_subject,
        quantile_cont(c.n_admissions, 0.75) AS q3_admissions_per_subject,
        MAX(c.n_admissions) AS max_admissions_per_subject
    FROM control_subject_admission_counts c
    JOIN patients p
        ON c.subject_id = p.subject_id
)
SELECT * FROM case_summary
UNION ALL
SELECT * FROM control_summary;

WITH case_cohort AS (
    SELECT
        b.subject_id,
        b.hadm_id
    FROM base_admissions_mhc_subject_groups b
    WHERE b.subject_group = 'only_MHC1'
      AND b.has_secondary_psychiatric_same_admission = 1
),
control_cohort AS (
    SELECT
        b.subject_id,
        b.hadm_id
    FROM base_admissions_mhc_subject_groups b
    WHERE b.subject_group = 'only_MHC0'
),
case_subject_admission_counts AS (
    SELECT
        'Case (MHC=1 same-admission only)' AS group_name,
        subject_id,
        COUNT(DISTINCT hadm_id) AS n_admissions
    FROM case_cohort
    GROUP BY subject_id
),
control_subject_admission_counts AS (
    SELECT
        'Control (MHC=0)' AS group_name,
        subject_id,
        COUNT(DISTINCT hadm_id) AS n_admissions
    FROM control_cohort
    GROUP BY subject_id
)
SELECT * FROM case_subject_admission_counts
UNION ALL
SELECT * FROM control_subject_admission_counts;

WITH case_subjects AS (
    SELECT DISTINCT
        b.subject_id,
        'Case (MHC1-sa)' AS group_name
    FROM base_admissions_mhc_subject_groups b
    WHERE b.subject_group = 'only_MHC1'
      AND b.has_secondary_psychiatric_same_admission = 1
),
control_subjects AS (
    SELECT DISTINCT
        b.subject_id,
        'Control (MHC0)' AS group_name
    FROM base_admissions_mhc_subject_groups b
    WHERE b.subject_group = 'only_MHC0'
)
SELECT
    s.group_name,
    s.subject_id,
    p.anchor_age
FROM (
    SELECT * FROM case_subjects
    UNION ALL
    SELECT * FROM control_subjects
) s
JOIN patients p
    ON s.subject_id = p.subject_id
ORDER BY s.group_name, s.subject_id;