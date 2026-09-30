-- Finalized cohort analysis: stage 7 - matched cohort characterization
--
-- Prerequisites:
--   1. Run 01 through 04 in this folder.
--   2. Complete chief-complaint preprocessing and admission-level matching.
--   3. Run 06-Import_matched_cohort_and_export_additional_information.sql.
--
-- All analyses in this script use the final matched analytic cohort. The
-- MHC1-psychosis denominator is therefore the matched case admissions, not the
-- full pre-matching eligible pool.


-- ---------------------------------------------------------------------------
-- 1. Matched-cohort integrity and psychosis-context composition
-- ---------------------------------------------------------------------------

SELECT
    cohort,
    matched_role,
    COUNT(*) AS n_rows,
    COUNT(DISTINCT pair_id) AS n_pairs,
    COUNT(DISTINCT subject_id) AS n_subjects,
    COUNT(DISTINCT hadm_id) AS n_admissions
FROM matched_cohort
GROUP BY cohort, matched_role
ORDER BY matched_role, cohort;


CREATE OR REPLACE TABLE matched_MHC1_psychosis_admissions AS
SELECT
    mc.pair_id,
    mc.matched_role,
    mc.cohort,
    mc.subject_id,
    mc.hadm_id,
    m.admittime,
    m.has_prior_psychiatric_history,
    m.has_current_secondary_psychiatric_icd,
    m.mhc1_version,
    m.has_prior_psychosis,
    m.has_current_secondary_psychosis,
    m.psychosis_context_version
FROM matched_cohort mc
JOIN finalized_MHC1_psychosis_excluding_over_50_MIMIC_admissions m
    ON mc.subject_id = m.subject_id
   AND mc.hadm_id = m.hadm_id
WHERE mc.cohort = 'MHC1_psychotic'
  AND mc.matched_role = 'case';


-- Expected: 6,432 rows, 6,432 admissions, and 6,432 pairs.
SELECT
    COUNT(*) AS n_rows,
    COUNT(DISTINCT pair_id) AS n_pairs,
    COUNT(DISTINCT subject_id) AS n_subjects,
    COUNT(DISTINCT hadm_id) AS n_admissions
FROM matched_MHC1_psychosis_admissions;


-- These context versions are mutually exclusive at admission level. Subjects
-- can contribute admissions to more than one version, so subject counts are
-- not additive across rows.
SELECT
    psychosis_context_version,
    COUNT(DISTINCT subject_id) AS n_subjects,
    ROUND(
        100.0 * COUNT(DISTINCT subject_id)
        / NULLIF((SELECT COUNT(DISTINCT subject_id)
                  FROM matched_MHC1_psychosis_admissions), 0),
        2
    ) AS pct_matched_MHC1_subjects,
    COUNT(DISTINCT hadm_id) AS n_admissions,
    ROUND(
        100.0 * COUNT(DISTINCT hadm_id)
        / NULLIF((SELECT COUNT(DISTINCT hadm_id)
                  FROM matched_MHC1_psychosis_admissions), 0),
        2
    ) AS pct_matched_MHC1_admissions
FROM matched_MHC1_psychosis_admissions
GROUP BY psychosis_context_version
ORDER BY n_admissions DESC, psychosis_context_version;


-- ---------------------------------------------------------------------------
-- 2. Mutually exclusive psychiatric ICD-code categories
-- ---------------------------------------------------------------------------

-- Mutual exclusivity applies to ICD codes, not to admissions or subjects.
-- Psychosis is evaluated first so affective and postpartum psychosis codes do
-- not simultaneously create a nonpsychotic comorbidity category.
CREATE OR REPLACE TABLE matched_psychiatric_icd_subcategories AS
SELECT DISTINCT
    p.icd_version,
    p.icd_code,
    p.long_title,
    CASE
        WHEN psy.icd_code IS NOT NULL THEN 'psychotic'

        WHEN
            (p.icd_version = 10 AND (
                p.icd_code LIKE 'F10%' OR p.icd_code LIKE 'F11%' OR
                p.icd_code LIKE 'F12%' OR p.icd_code LIKE 'F13%' OR
                p.icd_code LIKE 'F14%' OR p.icd_code LIKE 'F15%' OR
                p.icd_code LIKE 'F16%' OR p.icd_code LIKE 'F17%' OR
                p.icd_code LIKE 'F18%' OR p.icd_code LIKE 'F19%'
            ))
            OR
            (p.icd_version = 9 AND (
                p.icd_code LIKE '291%' OR p.icd_code LIKE '292%' OR
                p.icd_code LIKE '303%' OR p.icd_code LIKE '304%' OR
                p.icd_code LIKE '305%' OR p.icd_code = 'V113'
            ))
            THEN 'substance_related'

        WHEN
            (p.icd_version = 10 AND (
                p.icd_code LIKE 'F30%' OR p.icd_code LIKE 'F31%' OR
                p.icd_code LIKE 'F32%' OR p.icd_code LIKE 'F33%' OR
                p.icd_code LIKE 'F34%' OR p.icd_code LIKE 'F38%' OR
                p.icd_code LIKE 'F39%' OR p.icd_code LIKE 'F40%' OR
                p.icd_code LIKE 'F41%' OR p.icd_code LIKE 'F42%' OR
                p.icd_code LIKE 'F43%' OR p.icd_code LIKE 'F44%' OR
                p.icd_code LIKE 'F45%' OR p.icd_code LIKE 'F48%' OR
                p.icd_code LIKE 'F50%'
            ))
            OR
            (p.icd_version = 9 AND (
                p.icd_code LIKE '296%' OR p.icd_code LIKE '300%' OR
                p.icd_code LIKE '306%' OR p.icd_code LIKE '308%' OR
                p.icd_code LIKE '309%' OR p.icd_code = '311' OR
                p.icd_code LIKE '3071%' OR p.icd_code = '3130' OR
                p.icd_code = '3131' OR p.icd_code LIKE '3132%' OR
                p.icd_code = 'V111' OR p.icd_code = 'V112' OR
                p.icd_code = 'V114'
            ))
            THEN 'internalizing'

        WHEN
            (p.icd_version = 10 AND (
                p.icd_code LIKE 'F60%' OR p.icd_code LIKE 'F61%' OR
                p.icd_code LIKE 'F62%' OR p.icd_code LIKE 'F63%' OR
                p.icd_code LIKE 'F64%' OR p.icd_code LIKE 'F65%' OR
                p.icd_code LIKE 'F66%' OR p.icd_code LIKE 'F68%' OR
                p.icd_code LIKE 'F69%' OR p.icd_code LIKE 'F51%' OR
                p.icd_code LIKE 'F52%' OR p.icd_code LIKE 'F53%' OR
                p.icd_code LIKE 'F54%' OR p.icd_code LIKE 'F55%' OR
                p.icd_code LIKE 'F59%'
            ))
            OR
            (p.icd_version = 9 AND (
                p.icd_code LIKE '301%' OR p.icd_code LIKE '302%' OR
                p.icd_code LIKE '3070%' OR p.icd_code LIKE '3073%' OR
                p.icd_code LIKE '3074%' OR p.icd_code LIKE '3075%' OR
                p.icd_code LIKE '3076%' OR p.icd_code LIKE '3077%' OR
                p.icd_code LIKE '3078%' OR p.icd_code LIKE '3079%' OR
                p.icd_code LIKE '312%' OR p.icd_code LIKE '3133%' OR
                p.icd_code LIKE '3138%' OR p.icd_code = '3139'
            ))
            THEN 'personality_behavioral'

        WHEN
            (p.icd_version = 10 AND (
                p.icd_code LIKE 'F70%' OR p.icd_code LIKE 'F71%' OR
                p.icd_code LIKE 'F72%' OR p.icd_code LIKE 'F73%' OR
                p.icd_code LIKE 'F78%' OR p.icd_code LIKE 'F79%' OR
                p.icd_code LIKE 'F80%' OR p.icd_code LIKE 'F81%' OR
                p.icd_code LIKE 'F82%' OR p.icd_code LIKE 'F83%' OR
                p.icd_code LIKE 'F84%' OR p.icd_code LIKE 'F88%' OR
                p.icd_code LIKE 'F89%' OR p.icd_code LIKE 'F90%' OR
                p.icd_code LIKE 'F91%' OR p.icd_code LIKE 'F92%' OR
                p.icd_code LIKE 'F93%' OR p.icd_code LIKE 'F94%' OR
                p.icd_code LIKE 'F95%' OR p.icd_code LIKE 'F98%'
            ))
            OR
            (p.icd_version = 9 AND (
                p.icd_code LIKE '299%' OR p.icd_code LIKE '3072%' OR
                p.icd_code LIKE '314%' OR p.icd_code LIKE '315%' OR
                p.icd_code LIKE '317%' OR p.icd_code LIKE '318%' OR
                p.icd_code LIKE '319%'
            ))
            THEN 'neurodevelopmental'

        WHEN
            (p.icd_version = 10 AND (
                p.icd_code LIKE 'F01%' OR p.icd_code LIKE 'F02%' OR
                p.icd_code LIKE 'F03%'
            ))
            OR
            (p.icd_version = 9 AND (
                p.icd_code LIKE '290%' OR p.icd_code LIKE '2941%' OR
                p.icd_code LIKE '2942%' OR p.icd_code = '2948' OR
                p.icd_code = '2949'
            ))
            THEN 'neurocognitive'

        WHEN lower(p.long_title) LIKE '%suicid%'
          OR lower(p.long_title) LIKE '%self-harm%'
          OR lower(p.long_title) LIKE '%self harm%'
          OR lower(p.long_title) LIKE '%intentional self%'
            THEN 'suicide_self_harm'

        ELSE 'other'
    END AS psych_category
FROM psychiatric_icd_codes p
LEFT JOIN finalized_psychosis_icd_codes_extended psy
    ON p.icd_version = psy.icd_version
   AND p.icd_code = psy.icd_code;


-- Expected result: zero rows.
SELECT
    icd_version,
    icd_code,
    COUNT(*) AS n_category_rows
FROM matched_psychiatric_icd_subcategories
GROUP BY icd_version, icd_code
HAVING COUNT(*) > 1;


-- ---------------------------------------------------------------------------
-- 3. Nonpsychotic psychiatric comorbidity in matched MHC1-psychosis
-- ---------------------------------------------------------------------------

-- One row per matched MHC1 admission and nonpsychotic psychiatric category.
-- Prior and current-secondary flags are retained separately.
CREATE OR REPLACE TABLE matched_MHC1_psychosis_comorbidity AS
WITH comorbidity_context AS (
    SELECT DISTINCT
        m.subject_id,
        m.hadm_id,
        c.psych_category,
        0 AS has_prior_comorbidity,
        1 AS has_current_secondary_comorbidity
    FROM matched_MHC1_psychosis_admissions m
    JOIN diagnoses_icd d
        ON m.subject_id = d.subject_id
       AND m.hadm_id = d.hadm_id
    JOIN matched_psychiatric_icd_subcategories c
        ON d.icd_version = c.icd_version
       AND d.icd_code = c.icd_code
    WHERE d.seq_num > 1
      AND c.psych_category <> 'psychotic'

    UNION ALL

    SELECT DISTINCT
        m.subject_id,
        m.hadm_id,
        c.psych_category,
        1 AS has_prior_comorbidity,
        0 AS has_current_secondary_comorbidity
    FROM matched_MHC1_psychosis_admissions m
    JOIN admissions a_previous
        ON m.subject_id = a_previous.subject_id
       AND a_previous.hadm_id <> m.hadm_id
       AND a_previous.admittime < m.admittime
    JOIN diagnoses_icd d_previous
        ON a_previous.subject_id = d_previous.subject_id
       AND a_previous.hadm_id = d_previous.hadm_id
    JOIN matched_psychiatric_icd_subcategories c
        ON d_previous.icd_version = c.icd_version
       AND d_previous.icd_code = c.icd_code
    WHERE c.psych_category <> 'psychotic'
)
SELECT
    subject_id,
    hadm_id,
    psych_category AS comorbidity_category,
    MAX(has_prior_comorbidity) AS has_prior_comorbidity,
    MAX(has_current_secondary_comorbidity)
        AS has_current_secondary_comorbidity
FROM comorbidity_context
GROUP BY subject_id, hadm_id, psych_category;


-- Category frequencies. Categories are not additive because one admission or
-- subject may have multiple genuinely different comorbidities.
SELECT
    c.comorbidity_category,
    COUNT(DISTINCT c.subject_id) AS n_subjects,
    ROUND(
        100.0 * COUNT(DISTINCT c.subject_id)
        / NULLIF((SELECT COUNT(DISTINCT subject_id)
                  FROM matched_MHC1_psychosis_admissions), 0),
        2
    ) AS pct_matched_MHC1_subjects,
    COUNT(DISTINCT c.hadm_id) AS n_admissions,
    ROUND(
        100.0 * COUNT(DISTINCT c.hadm_id)
        / NULLIF((SELECT COUNT(DISTINCT hadm_id)
                  FROM matched_MHC1_psychosis_admissions), 0),
        2
    ) AS pct_matched_MHC1_admissions
FROM matched_MHC1_psychosis_comorbidity c
GROUP BY c.comorbidity_category
ORDER BY n_admissions DESC, c.comorbidity_category;


-- Prior versus current-secondary source for each category.
SELECT
    comorbidity_category,
    CASE
        WHEN has_prior_comorbidity = 1
         AND has_current_secondary_comorbidity = 1 THEN 'prior_and_current'
        WHEN has_prior_comorbidity = 1 THEN 'prior_only'
        ELSE 'current_secondary_only'
    END AS comorbidity_context,
    COUNT(DISTINCT subject_id) AS n_subjects,
    COUNT(DISTINCT hadm_id) AS n_admissions
FROM matched_MHC1_psychosis_comorbidity
GROUP BY comorbidity_category, comorbidity_context
ORDER BY comorbidity_category, comorbidity_context;


-- Number of distinct nonpsychotic comorbidity categories per admission,
-- including admissions with no identified comorbidity.
WITH category_counts AS (
    SELECT
        m.subject_id,
        m.hadm_id,
        COUNT(DISTINCT c.comorbidity_category) AS n_comorbidity_categories
    FROM matched_MHC1_psychosis_admissions m
    LEFT JOIN matched_MHC1_psychosis_comorbidity c
        ON m.subject_id = c.subject_id
       AND m.hadm_id = c.hadm_id
    GROUP BY m.subject_id, m.hadm_id
)
SELECT
    n_comorbidity_categories,
    COUNT(DISTINCT subject_id) AS n_subjects,
    COUNT(DISTINCT hadm_id) AS n_admissions
FROM category_counts
GROUP BY n_comorbidity_categories
ORDER BY n_comorbidity_categories;


-- ---------------------------------------------------------------------------
-- 4. Admission-level readmission and prior-utilization profile
-- ---------------------------------------------------------------------------

CREATE OR REPLACE TABLE matched_cohort_admission_utilization AS
WITH matched_index AS (
    SELECT
        mc.pair_id,
        mc.matched_role,
        mc.cohort,
        mc.subject_id,
        mc.hadm_id,
        a.admittime,
        a.dischtime
    FROM matched_cohort mc
    JOIN admissions a
        ON mc.subject_id = a.subject_id
       AND mc.hadm_id = a.hadm_id
),
prior_stats AS (
    SELECT
        i.pair_id,
        i.matched_role,
        i.cohort,
        i.subject_id,
        i.hadm_id,
        COUNT(DISTINCT p.hadm_id) AS n_prior_MIMIC_admissions
    FROM matched_index i
    LEFT JOIN admissions p
        ON i.subject_id = p.subject_id
       AND p.admittime < i.admittime
    GROUP BY i.pair_id, i.matched_role, i.cohort, i.subject_id, i.hadm_id
),
subsequent_stats AS (
    SELECT
        i.pair_id,
        i.matched_role,
        i.cohort,
        i.subject_id,
        i.hadm_id,
        COUNT(DISTINCT n.hadm_id) AS n_subsequent_MIMIC_admissions,
        COUNT(DISTINCT CASE
            WHEN n.admittime <= i.dischtime + INTERVAL '30 days'
            THEN n.hadm_id END) AS n_readmissions_within_30d,
        COUNT(DISTINCT CASE
            WHEN n.admittime <= i.dischtime + INTERVAL '90 days'
            THEN n.hadm_id END) AS n_readmissions_within_90d,
        COUNT(DISTINCT CASE
            WHEN n.admittime <= i.dischtime + INTERVAL '365 days'
            THEN n.hadm_id END) AS n_readmissions_within_365d,
        MIN(DATE_DIFF('day', i.dischtime, n.admittime))
            AS days_to_next_MIMIC_admission
    FROM matched_index i
    LEFT JOIN admissions n
        ON i.subject_id = n.subject_id
       AND n.admittime > i.dischtime
    GROUP BY i.pair_id, i.matched_role, i.cohort, i.subject_id, i.hadm_id
),
subject_totals AS (
    SELECT
        subject_id,
        COUNT(DISTINCT hadm_id) AS n_total_MIMIC_admissions
    FROM admissions
    GROUP BY subject_id
)
SELECT
    i.*,
    p.n_prior_MIMIC_admissions,
    s.n_subsequent_MIMIC_admissions,
    s.n_readmissions_within_30d,
    s.n_readmissions_within_90d,
    s.n_readmissions_within_365d,
    s.days_to_next_MIMIC_admission,
    CASE WHEN s.n_readmissions_within_30d > 0 THEN 1 ELSE 0 END
        AS has_readmission_within_30d,
    CASE WHEN s.n_readmissions_within_90d > 0 THEN 1 ELSE 0 END
        AS has_readmission_within_90d,
    CASE WHEN s.n_readmissions_within_365d > 0 THEN 1 ELSE 0 END
        AS has_readmission_within_365d,
    t.n_total_MIMIC_admissions
FROM matched_index i
JOIN prior_stats p
    ON i.pair_id = p.pair_id
   AND i.matched_role = p.matched_role
JOIN subsequent_stats s
    ON i.pair_id = s.pair_id
   AND i.matched_role = s.matched_role
JOIN subject_totals t
    ON i.subject_id = t.subject_id;


-- Admission-level readmission summary. Each matched admission is one
-- observation; repeated admissions from the same subject remain correlated.
SELECT
    cohort,
    matched_role,
    COUNT(DISTINCT subject_id) AS n_subjects,
    COUNT(DISTINCT hadm_id) AS n_matched_admissions,
    ROUND(AVG(n_prior_MIMIC_admissions), 2) AS mean_prior_admissions,
    MEDIAN(n_prior_MIMIC_admissions) AS median_prior_admissions,
    QUANTILE_CONT(n_prior_MIMIC_admissions, 0.25) AS q1_prior_admissions,
    QUANTILE_CONT(n_prior_MIMIC_admissions, 0.75) AS q3_prior_admissions,
    MAX(n_prior_MIMIC_admissions) AS max_prior_admissions,
    ROUND(100.0 * AVG((n_prior_MIMIC_admissions > 0)::INTEGER), 2)
        AS pct_with_any_prior_admission,
    ROUND(100.0 * AVG(has_readmission_within_30d), 2)
        AS pct_readmitted_within_30d,
    ROUND(100.0 * AVG(has_readmission_within_90d), 2)
        AS pct_readmitted_within_90d,
    ROUND(100.0 * AVG(has_readmission_within_365d), 2)
        AS pct_readmitted_within_365d,
    ROUND(AVG(n_total_MIMIC_admissions), 2)
        AS mean_total_MIMIC_admissions_for_subject,
    MEDIAN(n_total_MIMIC_admissions)
        AS median_total_MIMIC_admissions_for_subject,
    MAX(n_total_MIMIC_admissions)
        AS max_total_MIMIC_admissions_for_subject
FROM matched_cohort_admission_utilization
GROUP BY cohort, matched_role
ORDER BY matched_role, cohort;


-- ---------------------------------------------------------------------------
-- 5. Subject-level utilization profile
-- ---------------------------------------------------------------------------

CREATE OR REPLACE TABLE matched_cohort_subject_utilization AS
WITH matched_subjects AS (
    SELECT
        cohort,
        matched_role,
        subject_id,
        COUNT(DISTINCT hadm_id) AS n_matched_admissions
    FROM matched_cohort
    GROUP BY cohort, matched_role, subject_id
),
all_MIMIC_counts AS (
    SELECT
        subject_id,
        COUNT(DISTINCT hadm_id) AS n_total_MIMIC_admissions,
        COUNT(DISTINCT CASE
            WHEN upper(admission_type) LIKE '%EMER%'
              OR upper(admission_type) LIKE '%URGENT%'
            THEN hadm_id END) AS n_emergency_or_urgent_MIMIC_admissions
    FROM admissions
    GROUP BY subject_id
)
SELECT
    ms.*,
    a.n_total_MIMIC_admissions,
    a.n_emergency_or_urgent_MIMIC_admissions
FROM matched_subjects ms
JOIN all_MIMIC_counts a
    ON ms.subject_id = a.subject_id;


-- Each subject contributes once per cohort in this summary.
SELECT
    cohort,
    matched_role,
    COUNT(DISTINCT subject_id) AS n_subjects,
    SUM(n_matched_admissions) AS n_matched_admissions,
    ROUND(AVG(n_matched_admissions), 2) AS mean_matched_admissions_per_subject,
    MEDIAN(n_matched_admissions) AS median_matched_admissions_per_subject,
    QUANTILE_CONT(n_matched_admissions, 0.25)
        AS q1_matched_admissions_per_subject,
    QUANTILE_CONT(n_matched_admissions, 0.75)
        AS q3_matched_admissions_per_subject,
    MAX(n_matched_admissions) AS max_matched_admissions_per_subject,
    ROUND(AVG(n_total_MIMIC_admissions), 2)
        AS mean_total_MIMIC_admissions_per_subject,
    MEDIAN(n_total_MIMIC_admissions)
        AS median_total_MIMIC_admissions_per_subject,
    QUANTILE_CONT(n_total_MIMIC_admissions, 0.25)
        AS q1_total_MIMIC_admissions_per_subject,
    QUANTILE_CONT(n_total_MIMIC_admissions, 0.75)
        AS q3_total_MIMIC_admissions_per_subject,
    MAX(n_total_MIMIC_admissions) AS max_total_MIMIC_admissions_per_subject,
    ROUND(AVG(n_emergency_or_urgent_MIMIC_admissions), 2)
        AS mean_emergency_or_urgent_admissions_per_subject
FROM matched_cohort_subject_utilization
GROUP BY cohort, matched_role
ORDER BY matched_role, cohort;


-- Distribution of total MIMIC admissions per matched subject. This table makes
-- the utilization tail visible instead of reducing it to a single maximum.
SELECT
    cohort,
    matched_role,
    n_total_MIMIC_admissions,
    COUNT(DISTINCT subject_id) AS n_subjects,
    ROUND(
        100.0 * COUNT(DISTINCT subject_id)
        / SUM(COUNT(DISTINCT subject_id)) OVER (
            PARTITION BY cohort, matched_role
        ),
        2
    ) AS pct_subjects
FROM matched_cohort_subject_utilization
GROUP BY cohort, matched_role, n_total_MIMIC_admissions
ORDER BY matched_role, cohort, n_total_MIMIC_admissions;
