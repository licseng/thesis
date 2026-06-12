--Non-psychiatric main admissions
SELECT
    COUNT(DISTINCT a.hadm_id) AS n_primary_nonpsychiatric_admissions_with_discharge_and_primary_icd
FROM admissions a
JOIN discharge di
    ON a.subject_id = di.subject_id
   AND a.hadm_id = di.hadm_id
JOIN diagnoses_icd d1
    ON a.subject_id = d1.subject_id
   AND a.hadm_id = d1.hadm_id
   AND d1.seq_num = 1
WHERE NOT EXISTS (
    SELECT 1
    FROM psychiatric_icd_codes p
    WHERE p.icd_version = d1.icd_version
      AND p.icd_code = d1.icd_code
);

--Non-psychiatric AND excluding the grey-zone physical stuff admissions
SELECT
    COUNT(DISTINCT a.hadm_id) AS n_clean_primary_nonpsychiatric_admissions_with_discharge_and_primary_icd,
    COUNT(DISTINCT a.subject_id) AS n_subjects
FROM admissions a
JOIN discharge di
    ON a.subject_id = di.subject_id
   AND a.hadm_id = di.hadm_id
JOIN diagnoses_icd d1
    ON a.subject_id = d1.subject_id
   AND a.hadm_id = d1.hadm_id
   AND d1.seq_num = 1
WHERE NOT EXISTS (
    SELECT 1
    FROM psychiatric_icd_codes p
    WHERE p.icd_version = d1.icd_version
      AND p.icd_code = d1.icd_code
)
AND NOT EXISTS (
    SELECT 1
    FROM grey_zone_physical_icd_codes g
    WHERE g.icd_version = d1.icd_version
      AND g.icd_code = d1.icd_code
);
