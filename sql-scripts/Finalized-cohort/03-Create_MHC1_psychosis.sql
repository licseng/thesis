-- Finalized cohort construction: stage 3
--
-- Prerequisites:
--   Run 01-Load_and_create_base_tables.sql first.
--   Run 02-Create_MHC0_and_MHC1.sql second.
--
-- This script:
--   1. creates the active extended psychosis ICD definition;
--   2. carves MHC1-psychosis out of finalized_MHC1.
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
             AND d.icd_code IN ('F302', 'F312', 'F315', 'F3164')
            THEN 'bipolar_manic_with_psychosis'
        WHEN d.icd_version = 9
             AND d.icd_code IN ('29604', '29614', '29644', '29654', '29664')
            THEN 'bipolar_manic_with_psychosis'
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

