--1. Overlap patterns across the 4 focus groups in prior history
WITH admission_group_flags AS (
    SELECT
        p.subject_id,
        p.hadm_id,

        MAX(CASE WHEN lower(p.prior_psych_categories) LIKE '%internalizing%' THEN 1 ELSE 0 END) AS has_internalizing,
        MAX(CASE WHEN lower(p.prior_psych_categories) LIKE '%substance_related%' THEN 1 ELSE 0 END) AS has_substance_related,
        MAX(CASE WHEN lower(p.prior_psych_categories) LIKE '%psychotic%' THEN 1 ELSE 0 END) AS has_psychotic,
        MAX(CASE WHEN lower(p.prior_psych_categories) LIKE '%personality_behavioral%' THEN 1 ELSE 0 END) AS has_personality_behavioral

    FROM possible_overshadowing_admissions_psychiatric_MHH_focus4 p
    GROUP BY
        p.subject_id,
        p.hadm_id
),

admission_comorbidity_patterns AS (
    SELECT
        subject_id,
        hadm_id,
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

--2. Total admissions carrying any of the 4 focus groups in prior history, plus each focus group count
WITH focus4_counts AS (
    SELECT
        psych_category,
        COUNT(DISTINCT hadm_id) AS n_admissions
    FROM (
        SELECT hadm_id, 'internalizing' AS psych_category
        FROM possible_overshadowing_admissions_psychiatric_MHH_focus4
        WHERE lower(prior_psych_categories) LIKE '%internalizing%'

        UNION ALL

        SELECT hadm_id, 'substance_related'
        FROM possible_overshadowing_admissions_psychiatric_MHH_focus4
        WHERE lower(prior_psych_categories) LIKE '%substance_related%'

        UNION ALL

        SELECT hadm_id, 'psychotic'
        FROM possible_overshadowing_admissions_psychiatric_MHH_focus4
        WHERE lower(prior_psych_categories) LIKE '%psychotic%'

        UNION ALL

        SELECT hadm_id, 'personality_behavioral'
        FROM possible_overshadowing_admissions_psychiatric_MHH_focus4
        WHERE lower(prior_psych_categories) LIKE '%personality_behavioral%'
    ) x
    GROUP BY psych_category
)
SELECT
    'MHH_focus4_admissions' AS label,
    COUNT(DISTINCT hadm_id) AS n_admissions
FROM possible_overshadowing_admissions_psychiatric_MHH_focus4

UNION ALL

SELECT
    psych_category AS label,
    n_admissions
FROM focus4_counts;

--3. Number of prior-history psychiatric subgroup comorbidities within each focus group
WITH admission_all_psych_group_counts AS (
    SELECT
        subject_id,
        hadm_id,
        array_length(string_split(prior_psych_categories, ' | ')) AS n_all_prior_psych_groups
    FROM possible_overshadowing_admissions_psychiatric_MHH_focus4
    WHERE prior_psych_categories IS NOT NULL
),

focus_group_membership AS (
    SELECT DISTINCT
        subject_id,
        hadm_id,
        'internalizing' AS focus_group
    FROM possible_overshadowing_admissions_psychiatric_MHH_focus4
    WHERE lower(prior_psych_categories) LIKE '%internalizing%'

    UNION ALL

    SELECT DISTINCT
        subject_id,
        hadm_id,
        'substance_related'
    FROM possible_overshadowing_admissions_psychiatric_MHH_focus4
    WHERE lower(prior_psych_categories) LIKE '%substance_related%'

    UNION ALL

    SELECT DISTINCT
        subject_id,
        hadm_id,
        'psychotic'
    FROM possible_overshadowing_admissions_psychiatric_MHH_focus4
    WHERE lower(prior_psych_categories) LIKE '%psychotic%'

    UNION ALL

    SELECT DISTINCT
        subject_id,
        hadm_id,
        'personality_behavioral'
    FROM possible_overshadowing_admissions_psychiatric_MHH_focus4
    WHERE lower(prior_psych_categories) LIKE '%personality_behavioral%'
)

SELECT
    f.focus_group,
    a.n_all_prior_psych_groups,
    COUNT(DISTINCT f.hadm_id) AS n_admissions,
    COUNT(DISTINCT f.subject_id) AS n_subjects
FROM focus_group_membership f
JOIN admission_all_psych_group_counts a
    ON f.subject_id = a.subject_id
   AND f.hadm_id = a.hadm_id
GROUP BY
    f.focus_group,
    a.n_all_prior_psych_groups
ORDER BY
    f.focus_group,
    a.n_all_prior_psych_groups;

