--Size of the excluded (0,1) group
SELECT
    COUNT(DISTINCT hadm_id) AS n_admissions,
    COUNT(DISTINCT subject_id) AS n_subjects
FROM possible_overshadowing_admissions_psychiatric
WHERE has_secondary_psychiatric_same_admission = 0
  AND has_prior_psychiatric_history = 1;

-- Frequency of each prior-history subgroup in the excluded (0,1) group
WITH excluded_history_only AS (
    SELECT
        subject_id,
        hadm_id,
        prior_psych_categories
    FROM possible_overshadowing_admissions_psychiatric
    WHERE has_secondary_psychiatric_same_admission = 0
      AND has_prior_psychiatric_history = 1
),
split_categories AS (
    SELECT
        subject_id,
        hadm_id,
        trim(unnest(string_split(prior_psych_categories, ' | '))) AS psych_category
    FROM excluded_history_only
    WHERE prior_psych_categories IS NOT NULL
)
SELECT
    psych_category,
    COUNT(DISTINCT hadm_id) AS n_admissions,
    COUNT(DISTINCT subject_id) AS n_subjects
FROM split_categories
GROUP BY psych_category
ORDER BY n_admissions DESC, psych_category;

--Fractions of excluded admissions carrying each subgroup
WITH excluded_history_only AS (
    SELECT
        subject_id,
        hadm_id,
        prior_psych_categories
    FROM possible_overshadowing_admissions_psychiatric
    WHERE has_secondary_psychiatric_same_admission = 0
      AND has_prior_psychiatric_history = 1
),
split_categories AS (
    SELECT
        subject_id,
        hadm_id,
        trim(unnest(string_split(prior_psych_categories, ' | '))) AS psych_category
    FROM excluded_history_only
    WHERE prior_psych_categories IS NOT NULL
),
denom AS (
    SELECT COUNT(DISTINCT hadm_id) AS total_admissions
    FROM excluded_history_only
)
SELECT
    s.psych_category,
    COUNT(DISTINCT s.hadm_id) AS n_admissions,
    ROUND(100.0 * COUNT(DISTINCT s.hadm_id) / d.total_admissions, 2) AS pct_of_excluded_admissions
FROM split_categories s
CROSS JOIN denom d
GROUP BY s.psych_category, d.total_admissions
ORDER BY n_admissions DESC, s.psych_category;

--Number of prior-history subgroups per excluded admission
WITH excluded_history_only AS (
    SELECT
        subject_id,
        hadm_id,
        array_length(string_split(prior_psych_categories, ' | ')) AS n_prior_categories
    FROM possible_overshadowing_admissions_psychiatric
    WHERE has_secondary_psychiatric_same_admission = 0
      AND has_prior_psychiatric_history = 1
      AND prior_psych_categories IS NOT NULL
)
SELECT
    n_prior_categories,
    COUNT(DISTINCT hadm_id) AS n_admissions,
    COUNT(DISTINCT subject_id) AS n_subjects
FROM excluded_history_only
GROUP BY n_prior_categories
ORDER BY n_prior_categories;

--Most common exact prior-history combinations
SELECT
    prior_psych_categories,
    COUNT(DISTINCT hadm_id) AS n_admissions,
    COUNT(DISTINCT subject_id) AS n_subjects
FROM possible_overshadowing_admissions_psychiatric
WHERE has_secondary_psychiatric_same_admission = 0
  AND has_prior_psychiatric_history = 1
  AND prior_psych_categories IS NOT NULL
GROUP BY prior_psych_categories
ORDER BY n_admissions DESC, prior_psych_categories;
