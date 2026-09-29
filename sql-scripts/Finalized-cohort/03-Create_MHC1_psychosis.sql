-- Finalized cohort construction: stage 3
--
-- Prerequisites:
--   Run 01-Load_and_create_base_tables.sql first.
--   Run 02-Create_MHC0_and_MHC1.sql second.
--
-- This script:
--   1. creates the active extended psychosis ICD definition;
--   2. carves MHC1-psychosis out of finalized_MHC1; and
--   3. profiles admission counts among MHC1-psychosis subjects; and
--   4. excludes subjects with more than 50 total MIMIC-IV admissions from the
--      restricted MHC0 and MHC1-psychosis cohorts used by stage 4.
--
-- Temporal rule:
-- For a given physical admission, psychiatric information is counted only when
-- it occurs in an earlier hospital admission or as a secondary diagnosis in
-- the current physical admission. Diagnoses first recorded after the physical
-- admission are not carried backward.


-- ---------------------------------------------------------------------------
-- 1. Active extended psychosis ICD definition
-- ---------------------------------------------------------------------------

-- The validated core follows Deo et al. (Schizophrenia Bulletin, 2024;
-- doi:10.1093/schbul/sbae064). The project-specific extensions are kept
-- explicit so that their contribution can be inspected separately.
CREATE OR REPLACE TABLE finalized_psychosis_icd_codes_extended AS
SELECT DISTINCT
    d.icd_version,
    d.icd_code,
    d.long_title,
    CASE
        WHEN d.icd_version = 10 AND d.icd_code LIKE 'F20%'
            THEN 'schizophrenia'
        WHEN d.icd_version = 9 AND d.icd_code LIKE '2957%'
            THEN 'schizoaffective_disorder'
        WHEN d.icd_version = 9 AND d.icd_code LIKE '295%'
            THEN 'schizophrenia'
        WHEN d.icd_version = 10 AND d.icd_code LIKE 'F25%'
            THEN 'schizoaffective_disorder'
        WHEN d.icd_version = 10
             AND d.icd_code IN ('F302', 'F312')
            THEN 'bipolar_manic_with_psychosis'
        WHEN d.icd_version = 9
             AND d.icd_code IN ('29604', '29614', '29644')
            THEN 'bipolar_manic_with_psychosis'
        WHEN (d.icd_version = 10 AND d.icd_code = 'F315')
          OR (d.icd_version = 9 AND d.icd_code = '29654')
            THEN 'bipolar_depressed_with_psychosis'
        WHEN (d.icd_version = 10 AND d.icd_code = 'F3164')
          OR (d.icd_version = 9 AND d.icd_code = '29664')
            THEN 'bipolar_mixed_with_psychosis'
        WHEN d.icd_version = 10 AND d.icd_code IN ('F323', 'F333')
            THEN 'major_depression_with_psychosis'
        WHEN d.icd_version = 9 AND d.icd_code IN ('29624', '29634')
            THEN 'major_depression_with_psychosis'
        WHEN (d.icd_version = 10 AND d.icd_code LIKE 'F24%')
          OR (d.icd_version = 9 AND d.icd_code = '2973')
            THEN 'induced_shared_delusional_disorder'
        WHEN d.icd_version = 10 AND d.icd_code LIKE 'F531%'
            THEN 'postpartum_psychosis'
        WHEN d.icd_version = 9 AND d.icd_code = '2980'
            THEN 'depressive_type_psychosis'
        WHEN d.icd_version = 9 AND d.icd_code = 'V110'
            THEN 'historical_schizophrenia_status'
        WHEN (d.icd_version = 10 AND d.icd_code LIKE 'F22%')
          OR (d.icd_version = 9
              AND d.icd_code IN ('2970', '2971', '2972', '2978', '2979'))
            THEN 'delusional_paranoid_disorders'
        WHEN (d.icd_version = 10 AND d.icd_code LIKE 'F23%')
          OR (d.icd_version = 9
              AND d.icd_code IN ('2981', '2983', '2984', '2988'))
            THEN 'brief_acute_reactive_psychosis'
        WHEN d.icd_version = 10 AND d.icd_code LIKE 'F28%'
            THEN 'other_specified_psychotic_disorder'
        WHEN (d.icd_version = 10 AND d.icd_code LIKE 'F29%')
          OR (d.icd_version = 9 AND d.icd_code = '2989')
            THEN 'unspecified_psychosis'
        ELSE 'other_primary_psychotic_disorder'
    END AS diagnostic_group,
    CASE
        WHEN d.icd_version = 10
             AND (d.icd_code LIKE 'F24%' OR d.icd_code LIKE 'F531%')
            THEN 'clinical_extension'
        WHEN d.icd_version = 9 AND d.icd_code = '2980'
            THEN 'clinical_extension'
        WHEN d.icd_version = 9 AND d.icd_code = 'V110'
            THEN 'history_status'
        ELSE 'validated_core'
    END AS definition_tier
FROM d_icd_diagnoses d
WHERE
    (
        d.icd_version = 10
        AND (
            d.icd_code LIKE 'F20%' OR
            d.icd_code LIKE 'F22%' OR
            d.icd_code LIKE 'F23%' OR
            d.icd_code LIKE 'F24%' OR
            d.icd_code LIKE 'F25%' OR
            d.icd_code LIKE 'F28%' OR
            d.icd_code LIKE 'F29%' OR
            d.icd_code IN (
                'F302', 'F312', 'F315', 'F3164',
                'F323', 'F333', 'F531'
            )
        )
    )
    OR
    (
        d.icd_version = 9
        AND (
            d.icd_code LIKE '295%' OR
            d.icd_code LIKE '297%' OR
            d.icd_code IN (
                '2980', '2981', '2983', '2984', '2988', '2989',
                '29604', '29614', '29624', '29634',
                '29644', '29654', '29664', 'V110'
            )
        )
    );



-- ---------------------------------------------------------------------------
-- 2. MHC1-psychosis admission cohort
-- ---------------------------------------------------------------------------

CREATE OR REPLACE TABLE finalized_MHC1_psychosis AS
WITH psychosis_context AS (
    -- Psychosis recorded as a secondary diagnosis in the current physical
    -- admission.
    SELECT DISTINCT
        m.subject_id,
        m.hadm_id,
        0 AS has_prior_psychosis,
        1 AS has_current_secondary_psychosis
    FROM finalized_MHC1 m
    JOIN diagnoses_icd d
        ON m.subject_id = d.subject_id
       AND m.hadm_id = d.hadm_id
    JOIN finalized_psychosis_icd_codes_extended p
        ON d.icd_version = p.icd_version
       AND d.icd_code = p.icd_code
    WHERE d.seq_num > 1

    UNION ALL

    -- Psychosis recorded at any diagnosis position in an earlier admission.
    SELECT DISTINCT
        m.subject_id,
        m.hadm_id,
        1 AS has_prior_psychosis,
        0 AS has_current_secondary_psychosis
    FROM finalized_MHC1 m
    JOIN admissions a_previous
        ON m.subject_id = a_previous.subject_id
       AND a_previous.hadm_id <> m.hadm_id
       AND a_previous.admittime < m.admittime
    JOIN diagnoses_icd d_previous
        ON a_previous.subject_id = d_previous.subject_id
       AND a_previous.hadm_id = d_previous.hadm_id
    JOIN finalized_psychosis_icd_codes_extended p
        ON d_previous.icd_version = p.icd_version
       AND d_previous.icd_code = p.icd_code
),

psychosis_flags AS (
    SELECT
        subject_id,
        hadm_id,
        MAX(has_prior_psychosis) AS has_prior_psychosis,
        MAX(has_current_secondary_psychosis)
            AS has_current_secondary_psychosis
    FROM psychosis_context
    GROUP BY subject_id, hadm_id
)
SELECT
    m.*,
    f.has_prior_psychosis,
    f.has_current_secondary_psychosis,
    CASE
        WHEN f.has_prior_psychosis = 1
         AND f.has_current_secondary_psychosis = 1
            THEN 'prior_and_current_secondary_psychosis'
        WHEN f.has_prior_psychosis = 1
            THEN 'prior_psychosis_only'
        ELSE 'current_secondary_psychosis_only'
    END AS psychosis_context_version
FROM finalized_MHC1 m
JOIN psychosis_flags f
    ON m.subject_id = f.subject_id
   AND m.hadm_id = f.hadm_id;


-- ---------------------------------------------------------------------------
-- 3. Admission-count analysis for MHC1-psychosis subjects
-- ---------------------------------------------------------------------------

-- Three scopes are retained because they answer different questions:
--   * MHC1_psychosis: admissions actually entering the psychosis analysis;
--   * all_MHC1: all eligible MHC1 admissions belonging to those subjects; and
--   * all_MIMIC: every hospital admission for those subjects in MIMIC-IV.
--
-- This table is subject-level and diagnostic only. It does not modify
-- finalized_MHC1_psychosis.
CREATE OR REPLACE TABLE finalized_MHC1_psychosis_admission_count_profile AS
WITH psychosis_subjects AS (
    SELECT DISTINCT subject_id
    FROM finalized_MHC1_psychosis
),

psychosis_cohort_counts AS (
    SELECT
        subject_id,
        COUNT(DISTINCT hadm_id) AS n_MHC1_psychosis_admissions
    FROM finalized_MHC1_psychosis
    GROUP BY subject_id
),

all_MHC1_counts AS (
    SELECT
        m.subject_id,
        COUNT(DISTINCT m.hadm_id) AS n_all_MHC1_admissions
    FROM finalized_MHC1 m
    JOIN psychosis_subjects s
        ON m.subject_id = s.subject_id
    GROUP BY m.subject_id
),

all_MIMIC_counts AS (
    SELECT
        a.subject_id,
        COUNT(DISTINCT a.hadm_id) AS n_all_MIMIC_admissions
    FROM admissions a
    JOIN psychosis_subjects s
        ON a.subject_id = s.subject_id
    GROUP BY a.subject_id
)

SELECT
    p.subject_id,
    p.n_MHC1_psychosis_admissions,
    m.n_all_MHC1_admissions,
    a.n_all_MIMIC_admissions
FROM psychosis_cohort_counts p
JOIN all_MHC1_counts m
    ON p.subject_id = m.subject_id
JOIN all_MIMIC_counts a
    ON p.subject_id = a.subject_id;


-- Overall admission-count statistics in each scope.
WITH scope_counts AS (
    SELECT
        subject_id,
        'MHC1_psychosis_analysis_cohort' AS admission_scope,
        n_MHC1_psychosis_admissions AS n_admissions
    FROM finalized_MHC1_psychosis_admission_count_profile

    UNION ALL

    SELECT
        subject_id,
        'all_MHC1_admissions_for_psychosis_subjects' AS admission_scope,
        n_all_MHC1_admissions AS n_admissions
    FROM finalized_MHC1_psychosis_admission_count_profile

    UNION ALL

    SELECT
        subject_id,
        'all_MIMIC_admissions_for_psychosis_subjects' AS admission_scope,
        n_all_MIMIC_admissions AS n_admissions
    FROM finalized_MHC1_psychosis_admission_count_profile
)
SELECT
    admission_scope,
    COUNT(*) AS n_subjects,
    SUM(n_admissions) AS n_admissions,
    ROUND(AVG(n_admissions), 2) AS mean_admissions_per_subject,
    MEDIAN(n_admissions) AS median_admissions_per_subject,
    QUANTILE_CONT(n_admissions, 0.25) AS q1_admissions_per_subject,
    QUANTILE_CONT(n_admissions, 0.75) AS q3_admissions_per_subject,
    MAX(n_admissions) AS max_admissions_per_subject,
    SUM(n_admissions > 1) AS n_subjects_with_multiple_admissions,
    ROUND(100.0 * SUM(n_admissions > 1) / NULLIF(COUNT(*), 0), 2)
        AS pct_subjects_with_multiple_admissions
FROM scope_counts
GROUP BY admission_scope
ORDER BY admission_scope;


-- Exact distribution of admissions per subject for documenting the difference
-- between analysis-cohort admission counts and complete MIMIC utilization.
WITH scope_counts AS (
    SELECT
        'MHC1_psychosis_analysis_cohort' AS admission_scope,
        n_MHC1_psychosis_admissions AS n_admissions
    FROM finalized_MHC1_psychosis_admission_count_profile

    UNION ALL

    SELECT
        'all_MHC1_admissions_for_psychosis_subjects' AS admission_scope,
        n_all_MHC1_admissions AS n_admissions
    FROM finalized_MHC1_psychosis_admission_count_profile

    UNION ALL

    SELECT
        'all_MIMIC_admissions_for_psychosis_subjects' AS admission_scope,
        n_all_MIMIC_admissions AS n_admissions
    FROM finalized_MHC1_psychosis_admission_count_profile
)
SELECT
    admission_scope,
    n_admissions,
    COUNT(*) AS n_subjects,
    ROUND(
        100.0 * COUNT(*)
        / NULLIF(SUM(COUNT(*)) OVER (PARTITION BY admission_scope), 0),
        2
    ) AS pct_subjects
FROM scope_counts
GROUP BY admission_scope, n_admissions
ORDER BY admission_scope, n_admissions;


-- ---------------------------------------------------------------------------
-- 4. Exclusion of extreme hospital utilizers
-- ---------------------------------------------------------------------------

-- A subject is excluded when their complete MIMIC-IV hospital trajectory
-- contains more than 50 distinct admissions. This rounded threshold is just
-- above the observed MHC0 maximum of 48 admissions and restricts both cohorts
-- to the utilization range represented by the control population. It is based
-- on all rows in `admissions`, not only admissions eligible for MHC0 or
-- MHC1-psychosis. The unrestricted finalized cohorts remain available.
CREATE OR REPLACE TABLE finalized_MHC0_MHC1_psychosis_utilization_profile AS
WITH analysis_subjects AS (
    SELECT DISTINCT
        'MHC0' AS cohort,
        subject_id
    FROM finalized_MHC0

    UNION ALL

    SELECT DISTINCT
        'MHC1_psychosis' AS cohort,
        subject_id
    FROM finalized_MHC1_psychosis
),

analysis_admission_counts AS (
    SELECT
        'MHC0' AS cohort,
        subject_id,
        COUNT(DISTINCT hadm_id) AS n_analysis_admissions
    FROM finalized_MHC0
    GROUP BY subject_id

    UNION ALL

    SELECT
        'MHC1_psychosis' AS cohort,
        subject_id,
        COUNT(DISTINCT hadm_id) AS n_analysis_admissions
    FROM finalized_MHC1_psychosis
    GROUP BY subject_id
),

all_MIMIC_admission_counts AS (
    SELECT
        s.cohort,
        s.subject_id,
        COUNT(DISTINCT a.hadm_id) AS n_all_MIMIC_admissions
    FROM analysis_subjects s
    JOIN admissions a
        ON s.subject_id = a.subject_id
    GROUP BY s.cohort, s.subject_id
)

SELECT
    s.cohort,
    s.subject_id,
    c.n_analysis_admissions,
    a.n_all_MIMIC_admissions,
    a.n_all_MIMIC_admissions > 50 AS exclude_over_50_all_MIMIC_admissions
FROM analysis_subjects s
JOIN analysis_admission_counts c
    ON s.cohort = c.cohort
   AND s.subject_id = c.subject_id
JOIN all_MIMIC_admission_counts a
    ON s.cohort = a.cohort
   AND s.subject_id = a.subject_id;


-- Compare the utilization distributions before choosing a restriction. A
-- maximum is reported, but upper percentiles are more stable because a single
-- unusual control subject can determine the maximum.
SELECT
    cohort,
    COUNT(*) AS n_subjects,
    QUANTILE_DISC(n_analysis_admissions, 0.95)
        AS p95_analysis_admissions,
    QUANTILE_DISC(n_analysis_admissions, 0.99)
        AS p99_analysis_admissions,
    QUANTILE_DISC(n_analysis_admissions, 0.995)
        AS p995_analysis_admissions,
    MAX(n_analysis_admissions) AS max_analysis_admissions,
    QUANTILE_DISC(n_all_MIMIC_admissions, 0.95)
        AS p95_all_MIMIC_admissions,
    QUANTILE_DISC(n_all_MIMIC_admissions, 0.99)
        AS p99_all_MIMIC_admissions,
    QUANTILE_DISC(n_all_MIMIC_admissions, 0.995)
        AS p995_all_MIMIC_admissions,
    MAX(n_all_MIMIC_admissions) AS max_all_MIMIC_admissions
FROM finalized_MHC0_MHC1_psychosis_utilization_profile
GROUP BY cohort
ORDER BY cohort;


-- Separately named restricted cohorts preserve the unrestricted source tables
-- while providing the finalized inputs used by stage 4.
CREATE OR REPLACE TABLE finalized_MHC0_excluding_over_50_MIMIC_admissions AS
SELECT m.*
FROM finalized_MHC0 m
JOIN finalized_MHC0_MHC1_psychosis_utilization_profile p
    ON m.subject_id = p.subject_id
WHERE p.cohort = 'MHC0'
  AND NOT p.exclude_over_50_all_MIMIC_admissions;


CREATE OR REPLACE TABLE finalized_MHC1_psychosis_excluding_over_50_MIMIC_admissions AS
SELECT m.*
FROM finalized_MHC1_psychosis m
JOIN finalized_MHC0_MHC1_psychosis_utilization_profile p
    ON m.subject_id = p.subject_id
WHERE p.cohort = 'MHC1_psychosis'
  AND NOT p.exclude_over_50_all_MIMIC_admissions;


-- Before/after impact of excluding complete patient trajectories above the
-- threshold. `n_analysis_admissions_excluded` includes every analysis admission
-- belonging to an excluded subject, not merely admissions after the fiftieth.
SELECT
    cohort,
    COUNT(*) AS n_subjects_before_exclusion,
    SUM(n_analysis_admissions) AS n_analysis_admissions_before_exclusion,
    SUM(exclude_over_50_all_MIMIC_admissions)
        AS n_subjects_excluded,
    SUM(
        CASE
            WHEN exclude_over_50_all_MIMIC_admissions
                THEN n_analysis_admissions
            ELSE 0
        END
    ) AS n_analysis_admissions_excluded,
    SUM(NOT exclude_over_50_all_MIMIC_admissions)
        AS n_subjects_after_exclusion,
    SUM(
        CASE
            WHEN NOT exclude_over_50_all_MIMIC_admissions
                THEN n_analysis_admissions
            ELSE 0
        END
    ) AS n_analysis_admissions_after_exclusion,
    ROUND(
        100.0 * SUM(exclude_over_50_all_MIMIC_admissions)
        / NULLIF(COUNT(*), 0),
        2
    ) AS pct_subjects_excluded,
    ROUND(
        100.0 * SUM(
            CASE
                WHEN exclude_over_50_all_MIMIC_admissions
                    THEN n_analysis_admissions
                ELSE 0
            END
        )
        / NULLIF(SUM(n_analysis_admissions), 0),
        2
    ) AS pct_analysis_admissions_excluded
FROM finalized_MHC0_MHC1_psychosis_utilization_profile
GROUP BY cohort
ORDER BY cohort;


-- Inspect the upper-utilization tail without exposing clinical text.
SELECT
    cohort,
    subject_id,
    n_analysis_admissions,
    n_all_MIMIC_admissions
FROM finalized_MHC0_MHC1_psychosis_utilization_profile
WHERE exclude_over_50_all_MIMIC_admissions
ORDER BY cohort, n_all_MIMIC_admissions DESC, subject_id;
