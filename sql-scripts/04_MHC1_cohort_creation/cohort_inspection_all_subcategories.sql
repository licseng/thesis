-- Explores the admission-level structure of the original MHC1-all cohort by 
-- summarizing same-admission vs prior-history psychiatric context, subgroup frequencies, 
-- subgroup combinations, and overall psychiatric-category complexity.
SELECT
    has_secondary_psychiatric_same_admission,
    has_prior_psychiatric_history,
    COUNT(DISTINCT hadm_id) AS n_admissions
	FROM possible_overshadowing_admissions_psychiatric
GROUP BY 1,2
ORDER BY 1,2;

SELECT *
FROM possible_overshadowing_admissions_psychiatric
LIMIT 50;

--Frequency of each same-admission category
WITH split_categories AS (
    SELECT
        subject_id,
        hadm_id,
        trim(unnest(string_split(same_admission_psych_categories, ' | '))) AS psych_category
    FROM possible_overshadowing_admissions_psychiatric
    WHERE same_admission_psych_categories IS NOT NULL
)
SELECT
    psych_category,
    COUNT(DISTINCT hadm_id) AS n_admissions,
    COUNT(DISTINCT subject_id) AS n_subjects
FROM split_categories
GROUP BY psych_category
ORDER BY n_admissions DESC, psych_category;

--Number of same-admission categories per admission
WITH admission_category_counts AS (
    SELECT
        subject_id,
        hadm_id,
        array_length(string_split(same_admission_psych_categories, ' | ')) AS n_same_admission_categories
    FROM possible_overshadowing_admissions_psychiatric
    WHERE same_admission_psych_categories IS NOT NULL
)
SELECT
    n_same_admission_categories,
    COUNT(DISTINCT hadm_id) AS n_admissions,
    COUNT(DISTINCT subject_id) AS n_subjects
FROM admission_category_counts
GROUP BY n_same_admission_categories
ORDER BY n_same_admission_categories;

--Most common same-admission category combinations
SELECT
    same_admission_psych_categories,
    COUNT(DISTINCT hadm_id) AS n_admissions,
FROM possible_overshadowing_admissions_psychiatric
WHERE same_admission_psych_categories IS NOT NULL
GROUP BY same_admission_psych_categories
ORDER BY n_admissions DESC, same_admission_psych_categories;

WITH combos AS (
    SELECT
        'same_admission' AS source,
        same_admission_psych_categories AS psych_categories,
        hadm_id
    FROM possible_overshadowing_admissions_psychiatric
    WHERE same_admission_psych_categories IS NOT NULL

    UNION ALL

    SELECT
        'prior_history' AS source,
        prior_psych_categories AS psych_categories,
        hadm_id
    FROM possible_overshadowing_admissions_psychiatric
    WHERE prior_psych_categories IS NOT NULL
),
ranked AS (
    SELECT
        source,
        psych_categories,
        COUNT(DISTINCT hadm_id) AS n_admissions,
        ROW_NUMBER() OVER (
            PARTITION BY source
            ORDER BY COUNT(DISTINCT hadm_id) DESC, psych_categories
        ) AS rn
    FROM combos
    GROUP BY source, psych_categories
)
SELECT
    source,
    psych_categories,
    n_admissions
FROM ranked
WHERE rn <= 20
ORDER BY source, rn;

WITH all_categories AS (
    SELECT
        subject_id,
        hadm_id,
        trim(unnest(string_split(same_admission_psych_categories, ' | '))) AS psych_category
    FROM possible_overshadowing_admissions_psychiatric
    WHERE same_admission_psych_categories IS NOT NULL

    UNION

    SELECT
        subject_id,
        hadm_id,
        trim(unnest(string_split(prior_psych_categories, ' | '))) AS psych_category
    FROM possible_overshadowing_admissions_psychiatric
    WHERE prior_psych_categories IS NOT NULL
)
SELECT
    psych_category,
    COUNT(DISTINCT hadm_id) AS n_admissions,
    COUNT(DISTINCT subject_id) AS n_subjects
FROM all_categories
GROUP BY psych_category
ORDER BY n_admissions DESC, psych_category;

WITH all_categories AS (
    SELECT
        subject_id,
        hadm_id,
        trim(unnest(string_split(same_admission_psych_categories, ' | '))) AS psych_category
    FROM possible_overshadowing_admissions_psychiatric
    WHERE same_admission_psych_categories IS NOT NULL

    UNION

    SELECT
        subject_id,
        hadm_id,
        trim(unnest(string_split(prior_psych_categories, ' | '))) AS psych_category
    FROM possible_overshadowing_admissions_psychiatric
    WHERE prior_psych_categories IS NOT NULL
),
admission_category_counts AS (
    SELECT
        subject_id,
        hadm_id,
        COUNT(DISTINCT psych_category) AS n_all_categories
    FROM all_categories
    GROUP BY subject_id, hadm_id
)
SELECT
    n_all_categories,
    COUNT(DISTINCT hadm_id) AS n_admissions,
    COUNT(DISTINCT subject_id) AS n_subjects
FROM admission_category_counts
GROUP BY n_all_categories
ORDER BY n_all_categories;

WITH all_categories AS (
    SELECT
        subject_id,
        hadm_id,
        trim(unnest(string_split(same_admission_psych_categories, ' | '))) AS psych_category
    FROM possible_overshadowing_admissions_psychiatric
    WHERE same_admission_psych_categories IS NOT NULL

    UNION

    SELECT
        subject_id,
        hadm_id,
        trim(unnest(string_split(prior_psych_categories, ' | '))) AS psych_category
    FROM possible_overshadowing_admissions_psychiatric
    WHERE prior_psych_categories IS NOT NULL
),
combined AS (
    SELECT
        subject_id,
        hadm_id,
        string_agg(psych_category, ' | ' ORDER BY psych_category) AS all_psych_categories
    FROM (
        SELECT DISTINCT
            subject_id,
            hadm_id,
            psych_category
        FROM all_categories
    ) x
    GROUP BY subject_id, hadm_id
)
SELECT
    all_psych_categories,
    COUNT(DISTINCT hadm_id) AS n_admissions,
    COUNT(DISTINCT subject_id) AS n_subjects
FROM combined
GROUP BY all_psych_categories
ORDER BY n_admissions DESC, all_psych_categories;
