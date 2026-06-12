-- Describes the psychiatric composition of the pure MHC1-sa cohort by
--(1) overlap patterns across the four focus groups, 
--(2) total admissions carrying any of the four focus groups, and 
--(3) the number of same-admission psychiatric subgroup comorbidities within each focus group.
WITH admission_group_flags AS (
    SELECT
        p.subject_id,
        p.hadm_id,

        MAX(CASE WHEN i.icd_code IS NOT NULL THEN 1 ELSE 0 END) AS has_internalizing,
        MAX(CASE WHEN s.icd_code IS NOT NULL THEN 1 ELSE 0 END) AS has_substance_related,
        MAX(CASE WHEN psy.icd_code IS NOT NULL THEN 1 ELSE 0 END) AS has_psychotic,
        MAX(CASE WHEN pb.icd_code IS NOT NULL THEN 1 ELSE 0 END) AS has_personality_behavioral

    FROM possible_overshadowing_admissions_psychiatric_focus4 p
    JOIN diagnoses_icd d
        ON p.subject_id = d.subject_id
       AND p.hadm_id = d.hadm_id
       AND d.seq_num > 1

    LEFT JOIN psychiatric_icd_codes_internalizing i
        ON d.icd_version = i.icd_version
       AND d.icd_code = i.icd_code

    LEFT JOIN psychiatric_icd_codes_substance_related s
        ON d.icd_version = s.icd_version
       AND d.icd_code = s.icd_code

    LEFT JOIN psychiatric_icd_codes_psychotic psy
        ON d.icd_version = psy.icd_version
       AND d.icd_code = psy.icd_code

    LEFT JOIN psychiatric_icd_codes_personality_behavioral pb
        ON d.icd_version = pb.icd_version
       AND d.icd_code = pb.icd_code

    GROUP BY
        p.subject_id,
        p.hadm_id
),

admission_comorbidity_patterns AS (
    SELECT
        subject_id,
        hadm_id,
        has_internalizing,
        has_substance_related,
        has_psychotic,
        has_personality_behavioral,
        CASE
            WHEN has_internalizing = 1 AND has_substance_related = 0 AND has_psychotic = 0 AND has_personality_behavioral = 0
                THEN 'internalizing_only'
            WHEN has_internalizing = 0 AND has_substance_related = 1 AND has_psychotic = 0 AND has_personality_behavioral = 0
                THEN 'substance_related_only'
            WHEN has_internalizing = 0 AND has_substance_related = 0 AND has_psychotic = 1 AND has_personality_behavioral = 0
                THEN 'psychotic_only'
            WHEN has_internalizing = 0 AND has_substance_related = 0 AND has_psychotic = 0 AND has_personality_behavioral = 1
                THEN 'personality_behavioral_only'

            WHEN has_internalizing = 1 AND has_substance_related = 1 AND has_psychotic = 0 AND has_personality_behavioral = 0
                THEN 'internalizing_substance'
            WHEN has_internalizing = 1 AND has_substance_related = 0 AND has_psychotic = 1 AND has_personality_behavioral = 0
                THEN 'internalizing_psychotic'
            WHEN has_internalizing = 1 AND has_substance_related = 0 AND has_psychotic = 0 AND has_personality_behavioral = 1
                THEN 'internalizing_personality'
            WHEN has_internalizing = 0 AND has_substance_related = 1 AND has_psychotic = 1 AND has_personality_behavioral = 0
                THEN 'substance_psychotic'
            WHEN has_internalizing = 0 AND has_substance_related = 1 AND has_psychotic = 0 AND has_personality_behavioral = 1
                THEN 'substance_personality'
            WHEN has_internalizing = 0 AND has_substance_related = 0 AND has_psychotic = 1 AND has_personality_behavioral = 1
                THEN 'psychotic_personality'

            WHEN has_internalizing = 1 AND has_substance_related = 1 AND has_psychotic = 1 AND has_personality_behavioral = 0
                THEN 'internalizing_substance_psychotic'
            WHEN has_internalizing = 1 AND has_substance_related = 1 AND has_psychotic = 0 AND has_personality_behavioral = 1
                THEN 'internalizing_substance_personality'
            WHEN has_internalizing = 1 AND has_substance_related = 0 AND has_psychotic = 1 AND has_personality_behavioral = 1
                THEN 'internalizing_psychotic_personality'
            WHEN has_internalizing = 0 AND has_substance_related = 1 AND has_psychotic = 1 AND has_personality_behavioral = 1
                THEN 'substance_psychotic_personality'

            WHEN has_internalizing = 1 AND has_substance_related = 1 AND has_psychotic = 1 AND has_personality_behavioral = 1
                THEN 'all_four'

            ELSE 'none_of_focus4'
        END AS comorbidity_pattern
    FROM admission_group_flags
)

SELECT
    comorbidity_pattern,
    COUNT(DISTINCT hadm_id) AS n_admissions,
    COUNT(DISTINCT subject_id) AS n_subjects
FROM admission_comorbidity_patterns
GROUP BY comorbidity_pattern
ORDER BY n_admissions DESC, comorbidity_pattern;


WITH any_secondary_psych AS (
    SELECT
        COUNT(DISTINCT hadm_id) AS n_admissions
    FROM base_admissions_mhc_subject_groups
    WHERE subject_group = 'only_MHC1'
      AND has_secondary_psychiatric_same_admission = 1
),
focus4_counts AS (
    SELECT
        psych_category,
        COUNT(DISTINCT hadm_id) AS n_admissions
    FROM (
        SELECT b.hadm_id, 'internalizing' AS psych_category
        FROM base_admissions_mhc_subject_groups b
        JOIN diagnoses_icd d
            ON b.subject_id = d.subject_id
           AND b.hadm_id = d.hadm_id
        JOIN psychiatric_icd_codes_internalizing c
            ON d.icd_version = c.icd_version
           AND d.icd_code = c.icd_code
        WHERE b.subject_group = 'only_MHC1'
          AND d.seq_num > 1

        UNION ALL

        SELECT b.hadm_id, 'substance_related'
        FROM base_admissions_mhc_subject_groups b
        JOIN diagnoses_icd d
            ON b.subject_id = d.subject_id
           AND b.hadm_id = d.hadm_id
        JOIN psychiatric_icd_codes_substance_related c
            ON d.icd_version = c.icd_version
           AND d.icd_code = c.icd_code
        WHERE b.subject_group = 'only_MHC1'
          AND d.seq_num > 1

        UNION ALL

        SELECT b.hadm_id, 'psychotic'
        FROM base_admissions_mhc_subject_groups b
        JOIN diagnoses_icd d
            ON b.subject_id = d.subject_id
           AND b.hadm_id = d.hadm_id
        JOIN psychiatric_icd_codes_psychotic c
            ON d.icd_version = c.icd_version
           AND d.icd_code = c.icd_code
        WHERE b.subject_group = 'only_MHC1'
          AND d.seq_num > 1

        UNION ALL

        SELECT b.hadm_id, 'personality_behavioral'
        FROM base_admissions_mhc_subject_groups b
        JOIN diagnoses_icd d
            ON b.subject_id = d.subject_id
           AND b.hadm_id = d.hadm_id
        JOIN psychiatric_icd_codes_personality_behavioral c
            ON d.icd_version = c.icd_version
           AND d.icd_code = c.icd_code
        WHERE b.subject_group = 'only_MHC1'
          AND d.seq_num > 1
    ) x
    GROUP BY psych_category
)
SELECT
    'only_MHC1_admissions' AS label,
    COUNT(DISTINCT hadm_id) AS n_admissions
FROM base_admissions_mhc_subject_groups
WHERE subject_group = 'only_MHC1'

UNION ALL

SELECT
    'only_MHC1_with_any_secondary_psychiatric_code' AS label,
    n_admissions
FROM any_secondary_psych

UNION ALL

SELECT
    psych_category AS label,
    n_admissions
FROM focus4_counts;

WITH admission_all_psych_group_counts AS (
    SELECT
        x.subject_id,
        x.hadm_id,
        COUNT(DISTINCT psych_category) AS n_all_same_admission_psych_groups
    FROM (
        SELECT
            b.subject_id,
            b.hadm_id,
            'internalizing' AS psych_category
        FROM base_admissions_mhc_subject_groups b
        JOIN diagnoses_icd d
            ON b.subject_id = d.subject_id
           AND b.hadm_id = d.hadm_id
        JOIN psychiatric_icd_codes_internalizing c
            ON d.icd_version = c.icd_version
           AND d.icd_code = c.icd_code
        WHERE b.subject_group = 'only_MHC1'
          AND d.seq_num > 1

        UNION ALL

        SELECT
            b.subject_id,
            b.hadm_id,
            'substance_related'
        FROM base_admissions_mhc_subject_groups b
        JOIN diagnoses_icd d
            ON b.subject_id = d.subject_id
           AND b.hadm_id = d.hadm_id
        JOIN psychiatric_icd_codes_substance_related c
            ON d.icd_version = c.icd_version
           AND d.icd_code = c.icd_code
        WHERE b.subject_group = 'only_MHC1'
          AND d.seq_num > 1

        UNION ALL

        SELECT
            b.subject_id,
            b.hadm_id,
            'psychotic'
        FROM base_admissions_mhc_subject_groups b
        JOIN diagnoses_icd d
            ON b.subject_id = d.subject_id
           AND b.hadm_id = d.hadm_id
        JOIN psychiatric_icd_codes_psychotic c
            ON d.icd_version = c.icd_version
           AND d.icd_code = c.icd_code
        WHERE b.subject_group = 'only_MHC1'
          AND d.seq_num > 1

        UNION ALL

        SELECT
            b.subject_id,
            b.hadm_id,
            'personality_behavioral'
        FROM base_admissions_mhc_subject_groups b
        JOIN diagnoses_icd d
            ON b.subject_id = d.subject_id
           AND b.hadm_id = d.hadm_id
        JOIN psychiatric_icd_codes_personality_behavioral c
            ON d.icd_version = c.icd_version
           AND d.icd_code = c.icd_code
        WHERE b.subject_group = 'only_MHC1'
          AND d.seq_num > 1

        UNION ALL

        SELECT
            b.subject_id,
            b.hadm_id,
            'neurodevelopmental'
        FROM base_admissions_mhc_subject_groups b
        JOIN diagnoses_icd d
            ON b.subject_id = d.subject_id
           AND b.hadm_id = d.hadm_id
        JOIN psychiatric_icd_codes_neurodevelopmental c
            ON d.icd_version = c.icd_version
           AND d.icd_code = c.icd_code
        WHERE b.subject_group = 'only_MHC1'
          AND d.seq_num > 1

        UNION ALL

        SELECT
            b.subject_id,
            b.hadm_id,
            'neurocognitive'
        FROM base_admissions_mhc_subject_groups b
        JOIN diagnoses_icd d
            ON b.subject_id = d.subject_id
           AND b.hadm_id = d.hadm_id
        JOIN psychiatric_icd_codes_neurocognitive c
            ON d.icd_version = c.icd_version
           AND d.icd_code = c.icd_code
        WHERE b.subject_group = 'only_MHC1'
          AND d.seq_num > 1

        UNION ALL

        SELECT
            b.subject_id,
            b.hadm_id,
            'suicide_self_harm'
        FROM base_admissions_mhc_subject_groups b
        JOIN diagnoses_icd d
            ON b.subject_id = d.subject_id
           AND b.hadm_id = d.hadm_id
        JOIN psychiatric_icd_codes_suicide_self_harm c
            ON d.icd_version = c.icd_version
           AND d.icd_code = c.icd_code
        WHERE b.subject_group = 'only_MHC1'
          AND d.seq_num > 1

        UNION ALL

        SELECT
            b.subject_id,
            b.hadm_id,
            'other'
        FROM base_admissions_mhc_subject_groups b
        JOIN diagnoses_icd d
            ON b.subject_id = d.subject_id
           AND b.hadm_id = d.hadm_id
        JOIN psychiatric_icd_codes_other c
            ON d.icd_version = c.icd_version
           AND d.icd_code = c.icd_code
        WHERE b.subject_group = 'only_MHC1'
          AND d.seq_num > 1
    ) x
    GROUP BY x.subject_id, x.hadm_id
),

focus_group_membership AS (
    SELECT DISTINCT
        b.subject_id,
        b.hadm_id,
        'internalizing' AS focus_group
    FROM base_admissions_mhc_subject_groups b
    JOIN diagnoses_icd d
        ON b.subject_id = d.subject_id
       AND b.hadm_id = d.hadm_id
    JOIN psychiatric_icd_codes_internalizing c
        ON d.icd_version = c.icd_version
       AND d.icd_code = c.icd_code
    WHERE b.subject_group = 'only_MHC1'
      AND d.seq_num > 1

    UNION ALL

    SELECT DISTINCT
        b.subject_id,
        b.hadm_id,
        'substance_related'
    FROM base_admissions_mhc_subject_groups b
    JOIN diagnoses_icd d
        ON b.subject_id = d.subject_id
       AND b.hadm_id = d.hadm_id
    JOIN psychiatric_icd_codes_substance_related c
        ON d.icd_version = c.icd_version
       AND d.icd_code = c.icd_code
    WHERE b.subject_group = 'only_MHC1'
      AND d.seq_num > 1

    UNION ALL

    SELECT DISTINCT
        b.subject_id,
        b.hadm_id,
        'psychotic'
    FROM base_admissions_mhc_subject_groups b
    JOIN diagnoses_icd d
        ON b.subject_id = d.subject_id
       AND b.hadm_id = d.hadm_id
    JOIN psychiatric_icd_codes_psychotic c
        ON d.icd_version = c.icd_version
       AND d.icd_code = c.icd_code
    WHERE b.subject_group = 'only_MHC1'
      AND d.seq_num > 1

    UNION ALL

    SELECT DISTINCT
        b.subject_id,
        b.hadm_id,
        'personality_behavioral'
    FROM base_admissions_mhc_subject_groups b
    JOIN diagnoses_icd d
        ON b.subject_id = d.subject_id
       AND b.hadm_id = d.hadm_id
    JOIN psychiatric_icd_codes_personality_behavioral c
        ON d.icd_version = c.icd_version
       AND d.icd_code = c.icd_code
    WHERE b.subject_group = 'only_MHC1'
      AND d.seq_num > 1
)

SELECT
    f.focus_group,
    a.n_all_same_admission_psych_groups,
    COUNT(DISTINCT f.hadm_id) AS n_admissions,
    COUNT(DISTINCT f.subject_id) AS n_subjects
FROM focus_group_membership f
JOIN admission_all_psych_group_counts a
    ON f.subject_id = a.subject_id
   AND f.hadm_id = a.hadm_id
GROUP BY
    f.focus_group,
    a.n_all_same_admission_psych_groups
ORDER BY
    f.focus_group,
    a.n_all_same_admission_psych_groups;