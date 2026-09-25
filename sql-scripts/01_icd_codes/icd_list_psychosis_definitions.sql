-- Psychosis ICD definitions used in the project.
--
-- Both definitions are retained here for transparency and reproducibility:
--   1. psychosis_icd_codes_original: the definition previously embedded in
--      03-Final_icd_list_subcategories.sql.
--   2. psychosis_icd_codes_extended: the revised definition used by the
--      current subgroup and cohort analyses.


-- ---------------------------------------------------------------------------
-- 1. ORIGINAL PSYCHOSIS SUBGROUP DEFINITION
-- ---------------------------------------------------------------------------
-- This is restored from the Git version of
-- 03-Final_icd_list_subcategories.sql without changing its code criteria.

CREATE OR REPLACE TABLE psychosis_icd_codes_original AS
SELECT DISTINCT
    icd_version,
    icd_code,
    long_title
FROM psychiatric_icd_codes
WHERE
    (
        icd_version = 10
        AND (
            icd_code LIKE 'F20%' OR
            icd_code LIKE 'F21%' OR
            icd_code LIKE 'F22%' OR
            icd_code LIKE 'F23%' OR
            icd_code LIKE 'F24%' OR
            icd_code LIKE 'F25%' OR
            icd_code LIKE 'F28%' OR
            icd_code LIKE 'F29%'
        )
    )
    OR
    (
        icd_version = 9
        AND (
            icd_code LIKE '295%' OR
            icd_code LIKE '297%' OR
            icd_code LIKE '298%' OR
            icd_code = 'V110'
        )
    )
ORDER BY icd_version, icd_code;


-- ---------------------------------------------------------------------------
-- 2. EXTENDED PSYCHOTIC-ILLNESS DEFINITION (ACTIVE)
-- ---------------------------------------------------------------------------
-- Canonical ICD definition for a documented history of psychotic illness.
--
-- The core groups follow the code-based phenotype validated by Castro et al.
-- (Schizophrenia Bulletin, 2024; doi:10.1093/schbul/sbae064):
--   * schizophrenia
--   * schizoaffective disorder
--   * other primary psychotic disorders
--   * bipolar/manic disorders with psychosis
--   * major depressive disorder with psychosis
--
-- Two clinically justified extensions are retained as separate groups:
--   * F24: induced/shared delusional disorder (not substance-induced psychosis)
--   * F531: puerperal/postpartum psychosis
--
-- V110 is included because the MHH1 cohort is defined using documented
-- psychiatric history, and this code records a history of schizophrenia.
--
-- Intentionally excluded:
--   * F21 schizotypal disorder
--   * substance-induced psychosis (F10-F19 psychosis specifiers; ICD-9
--     2913, 2915, 29211, and 29212)
--   * psychosis due to a physiological condition (F060-F062; ICD-9
--     29381 and 29382)
--   * dementia with psychotic disturbance
--   * bipolar disorder or major depression without psychotic features

CREATE OR REPLACE TABLE psychosis_icd_codes_extended AS
SELECT DISTINCT
    d.icd_version,
    d.icd_code,
    d.long_title,
    CASE
        WHEN d.icd_version = 10 AND d.icd_code LIKE 'F20%'
            THEN 'schizophrenia'
        WHEN d.icd_version = 9
             AND d.icd_code LIKE '2957%'
            THEN 'schizoaffective_disorder'
        WHEN d.icd_version = 9
             AND d.icd_code LIKE '295%'
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
                'F302',
                'F312',
                'F315',
                'F3164',
                'F323',
                'F333',
                'F531'
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
                '2980',
                '2981',
                '2983',
                '2984',
                '2988',
                '2989',
                '29604',
                '29614',
                '29624',
                '29634',
                '29644',
                '29654',
                '29664',
                'V110'
            )
        )
    )
ORDER BY d.icd_version, d.icd_code;


-- Review the exact dictionary codes selected by each diagnostic group before
-- connecting this definition to cohort creation.
SELECT
    definition_tier,
    diagnostic_group,
    COUNT(*) AS n_codes
FROM psychosis_icd_codes_extended
GROUP BY definition_tier, diagnostic_group
ORDER BY definition_tier, diagnostic_group;


-- ---------------------------------------------------------------------------
-- SIDE-BY-SIDE COMPARISON
-- ---------------------------------------------------------------------------
-- This shows which dictionary codes were retained, added, or removed. The
-- revised definition is clinically broader, but is not a strict set superset
-- because F21 and some nonspecific ICD-9 298 codes are no longer included.

SELECT
    COALESCE(o.icd_version, e.icd_version) AS icd_version,
    COALESCE(o.icd_code, e.icd_code) AS icd_code,
    COALESCE(o.long_title, e.long_title) AS long_title,
    CASE WHEN o.icd_code IS NOT NULL THEN 1 ELSE 0 END AS in_original,
    CASE WHEN e.icd_code IS NOT NULL THEN 1 ELSE 0 END AS in_extended,
    CASE
        WHEN o.icd_code IS NOT NULL AND e.icd_code IS NOT NULL THEN 'retained'
        WHEN o.icd_code IS NULL THEN 'added_in_extended'
        ELSE 'removed_from_extended'
    END AS definition_change,
    e.diagnostic_group AS extended_diagnostic_group,
    e.definition_tier AS extended_definition_tier
FROM psychosis_icd_codes_original o
FULL OUTER JOIN psychosis_icd_codes_extended e
    ON o.icd_version = e.icd_version
   AND o.icd_code = e.icd_code
ORDER BY definition_change, icd_version, icd_code;
