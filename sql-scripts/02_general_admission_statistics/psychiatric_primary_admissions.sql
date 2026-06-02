--Total admissions
SELECT
    COUNT(*) AS n_admissions,
    COUNT(DISTINCT subject_id) AS n_subjects
FROM admissions;

--Admissions with discharge notes and a usable primary ICD diagnosis
SELECT
    COUNT(DISTINCT a.hadm_id) AS n_admissions_with_discharge_and_primary_icd,
    COUNT(DISTINCT a.subject_id) AS n_subjects_with_discharge_and_primary_icd
FROM admissions a
JOIN discharge d
    ON a.subject_id = d.subject_id
   AND a.hadm_id = d.hadm_id
JOIN diagnoses_icd d1
    ON a.subject_id = d1.subject_id
   AND a.hadm_id = d1.hadm_id
   AND d1.seq_num = 1;

--Counting only admissions with discharge notes where the PRIMARY diagnosis is the psychiatric subcategory
WITH primary_psychiatric_admissions AS (
    SELECT
        d.subject_id,
        d.hadm_id,
        CASE
            WHEN EXISTS (
                SELECT 1
                FROM psychiatric_icd_codes_psychotic p
                WHERE p.icd_version = d.icd_version
                  AND p.icd_code = d.icd_code
            ) THEN 'psychotic'

            WHEN EXISTS (
                SELECT 1
                FROM psychiatric_icd_codes_substance_related p
                WHERE p.icd_version = d.icd_version
                  AND p.icd_code = d.icd_code
            ) THEN 'substance_related'

            WHEN EXISTS (
                SELECT 1
                FROM psychiatric_icd_codes_internalizing p
                WHERE p.icd_version = d.icd_version
                  AND p.icd_code = d.icd_code
            ) THEN 'internalizing'

            WHEN EXISTS (
                SELECT 1
                FROM psychiatric_icd_codes_personality_behavioral p
                WHERE p.icd_version = d.icd_version
                  AND p.icd_code = d.icd_code
            ) THEN 'personality_behavioral'

            WHEN EXISTS (
                SELECT 1
                FROM psychiatric_icd_codes_neurodevelopmental p
                WHERE p.icd_version = d.icd_version
                  AND p.icd_code = d.icd_code
            ) THEN 'neurodevelopmental'

            WHEN EXISTS (
                SELECT 1
                FROM psychiatric_icd_codes_neurocognitive p
                WHERE p.icd_version = d.icd_version
                  AND p.icd_code = d.icd_code
            ) THEN 'neurocognitive'

            WHEN EXISTS (
                SELECT 1
                FROM psychiatric_icd_codes_suicide_self_harm p
                WHERE p.icd_version = d.icd_version
                  AND p.icd_code = d.icd_code
            ) THEN 'suicide_self_harm'

            WHEN EXISTS (
                SELECT 1
                FROM psychiatric_icd_codes_other p
                WHERE p.icd_version = d.icd_version
                  AND p.icd_code = d.icd_code
            ) THEN 'other'
        END AS psych_category
    FROM diagnoses_icd d
    JOIN discharge di
        ON d.subject_id = di.subject_id
       AND d.hadm_id = di.hadm_id
    WHERE d.seq_num = 1
      AND EXISTS (
          SELECT 1
          FROM psychiatric_icd_codes p
          WHERE p.icd_version = d.icd_version
            AND p.icd_code = d.icd_code
      )
)
SELECT
    psych_category,
    COUNT(DISTINCT hadm_id) AS n_admissions
FROM primary_psychiatric_admissions
GROUP BY psych_category

UNION ALL

SELECT
    'sum_of_subcategories' AS psych_category,
    COUNT(DISTINCT hadm_id) AS n_admissions
FROM primary_psychiatric_admissions

ORDER BY n_admissions DESC;