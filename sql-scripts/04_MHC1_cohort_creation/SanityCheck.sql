---- Validates key assumptions in the MHC framework by checking that mixed-group trajectories do not revert 
-- from MHC1 back to MHC0 and that same-admission category combinations sum correctly.
WITH admission_mhc AS (
    SELECT
        b.subject_id,
        b.hadm_id,
        b.admittime,
        CASE
            WHEN EXISTS (
                SELECT 1
                FROM admissions a_prev
                JOIN diagnoses_icd d_prev
                    ON a_prev.subject_id = d_prev.subject_id
                   AND a_prev.hadm_id = d_prev.hadm_id
                JOIN psychiatric_icd_codes p_prev
                    ON d_prev.icd_version = p_prev.icd_version
                   AND d_prev.icd_code = p_prev.icd_code
                WHERE a_prev.subject_id = b.subject_id
                  AND a_prev.admittime < b.admittime
            )
            OR EXISTS (
                SELECT 1
                FROM diagnoses_icd d_cur
                JOIN psychiatric_icd_codes p_cur
                    ON d_cur.icd_version = p_cur.icd_version
                   AND d_cur.icd_code = p_cur.icd_code
                WHERE d_cur.subject_id = b.subject_id
                  AND d_cur.hadm_id = b.hadm_id
                  AND d_cur.seq_num > 1
            )
            THEN 1
            ELSE 0
        END AS MHC
    FROM base_nonpsychiatric_admissions_with_discharge b
)

SELECT DISTINCT
    a1.subject_id
FROM admission_mhc a1
JOIN admission_mhc a2
    ON a1.subject_id = a2.subject_id
   AND a2.admittime > a1.admittime
WHERE a1.MHC = 1
  AND a2.MHC = 0
ORDER BY subject_id;

SELECT
    SUM(n_admissions) AS sum_of_combination_counts
FROM (
    SELECT
        same_admission_psych_categories,
        COUNT(DISTINCT hadm_id) AS n_admissions
    FROM possible_overshadowing_admissions_psychiatric
    WHERE same_admission_psych_categories IS NOT NULL
    GROUP BY same_admission_psych_categories
) x;

