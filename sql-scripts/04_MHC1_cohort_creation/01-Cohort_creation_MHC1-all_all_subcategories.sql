-- Table with admissions where there is mental health context (MHC = 1),
-- restricted to subject-exclusive case subjects (only_MHC1, no subjects with any MHC0 admissions)

CREATE OR REPLACE TABLE possible_overshadowing_admissions_psychiatric AS
WITH psychiatric_code_categories AS (
    SELECT icd_version, icd_code, 'psychotic' AS psych_category
    FROM psychiatric_icd_codes_psychotic

    UNION ALL
    SELECT icd_version, icd_code, 'substance_related'
    FROM psychiatric_icd_codes_substance_related

    UNION ALL
    SELECT icd_version, icd_code, 'internalizing'
    FROM psychiatric_icd_codes_internalizing

    UNION ALL
    SELECT icd_version, icd_code, 'personality_behavioral'
    FROM psychiatric_icd_codes_personality_behavioral

    UNION ALL
    SELECT icd_version, icd_code, 'neurodevelopmental'
    FROM psychiatric_icd_codes_neurodevelopmental

    UNION ALL
    SELECT icd_version, icd_code, 'neurocognitive'
    FROM psychiatric_icd_codes_neurocognitive

    UNION ALL
    SELECT icd_version, icd_code, 'suicide_self_harm'
    FROM psychiatric_icd_codes_suicide_self_harm

    UNION ALL
    SELECT icd_version, icd_code, 'other'
    FROM psychiatric_icd_codes_other
),

same_admission_categories AS (
    SELECT DISTINCT
        b.subject_id,
        b.hadm_id,
        c.psych_category
    FROM base_nonpsychiatric_admissions_with_discharge b
    JOIN diagnoses_icd d
        ON b.subject_id = d.subject_id
       AND b.hadm_id = d.hadm_id
    JOIN psychiatric_code_categories c
        ON d.icd_version = c.icd_version
       AND d.icd_code = c.icd_code
    WHERE d.seq_num > 1
),

same_admission_summary AS (
    SELECT
        subject_id,
        hadm_id,
        1 AS has_secondary_psychiatric_same_admission,
        string_agg(psych_category, ' | ' ORDER BY psych_category) AS same_admission_psych_categories
    FROM (
        SELECT DISTINCT
            subject_id,
            hadm_id,
            psych_category
        FROM same_admission_categories
    ) x
    GROUP BY subject_id, hadm_id
),

prior_history_categories AS (
    SELECT DISTINCT
        b.subject_id,
        b.hadm_id,
        c.psych_category
    FROM base_nonpsychiatric_admissions_with_discharge b
    JOIN admissions a_base
        ON b.subject_id = a_base.subject_id
       AND b.hadm_id = a_base.hadm_id
    JOIN admissions a_prev
        ON b.subject_id = a_prev.subject_id
       AND a_prev.admittime < a_base.admittime
    JOIN diagnoses_icd d_prev
        ON a_prev.subject_id = d_prev.subject_id
       AND a_prev.hadm_id = d_prev.hadm_id
    JOIN psychiatric_code_categories c
        ON d_prev.icd_version = c.icd_version
       AND d_prev.icd_code = c.icd_code
),

prior_history_summary AS (
    SELECT
        subject_id,
        hadm_id,
        1 AS has_prior_psychiatric_history,
        string_agg(psych_category, ' | ' ORDER BY psych_category) AS prior_psych_categories
    FROM (
        SELECT DISTINCT
            subject_id,
            hadm_id,
            psych_category
        FROM prior_history_categories
    ) x
    GROUP BY subject_id, hadm_id
),

admission_mhc AS (
    SELECT
        b.subject_id,
        b.hadm_id,
        b.admittime,
        COALESCE(s.has_secondary_psychiatric_same_admission, 0) AS has_secondary_psychiatric_same_admission,
        COALESCE(p.has_prior_psychiatric_history, 0) AS has_prior_psychiatric_history,
        s.same_admission_psych_categories,
        p.prior_psych_categories,
        CASE
            WHEN s.subject_id IS NOT NULL OR p.subject_id IS NOT NULL THEN 1
            ELSE 0
        END AS MHC
    FROM base_nonpsychiatric_admissions_with_discharge b
    LEFT JOIN same_admission_summary s
        ON b.subject_id = s.subject_id
       AND b.hadm_id = s.hadm_id
    LEFT JOIN prior_history_summary p
        ON b.subject_id = p.subject_id
       AND b.hadm_id = p.hadm_id
),

subject_groups AS (
    SELECT
        subject_id,
        MAX(CASE WHEN MHC = 1 THEN 1 ELSE 0 END) AS has_mhc1,
        MAX(CASE WHEN MHC = 0 THEN 1 ELSE 0 END) AS has_mhc0
    FROM admission_mhc
    GROUP BY subject_id
),

only_mhc1_subjects AS (
    SELECT subject_id
    FROM subject_groups
    WHERE has_mhc1 = 1
      AND has_mhc0 = 0
)

SELECT
    a.subject_id,
    a.hadm_id,
    a.admittime,
    a.has_secondary_psychiatric_same_admission,
    a.has_prior_psychiatric_history,
    a.same_admission_psych_categories,
    a.prior_psych_categories
FROM admission_mhc a
JOIN only_mhc1_subjects s
    ON a.subject_id = s.subject_id
WHERE a.MHC = 1;


