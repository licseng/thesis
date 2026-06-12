CREATE OR REPLACE TABLE possible_overshadowing_admissions_psychiatric_MHH_focus4 AS
SELECT DISTINCT
    p.*
FROM possible_overshadowing_admissions_psychiatric p
WHERE p.has_prior_psychiatric_history = 1;

SELECT
    COUNT(DISTINCT hadm_id) AS n_admissions,
    COUNT(DISTINCT subject_id) AS n_subjects
FROM possible_overshadowing_admissions_psychiatric_MHH_focus4;

--Inspect if classification is feasable
SELECT
    COUNT(DISTINCT p.hadm_id) AS n_history_only_admissions,
    COUNT(DISTINCT CASE
        WHEN regexp_matches(
            lower(d.text),
            'depress|anxiet|psychot|schizo|bipolar|ptsd|suicid|self-harm|self harm|hallucin|delusion|psychiatr|substance|alcohol|withdrawal'
        )
        THEN p.hadm_id
    END) AS n_with_psych_terms
FROM possible_overshadowing_admissions_psychiatric p
JOIN discharge d
    ON p.subject_id = d.subject_id
   AND p.hadm_id = d.hadm_id
WHERE p.has_prior_psychiatric_history = 1;