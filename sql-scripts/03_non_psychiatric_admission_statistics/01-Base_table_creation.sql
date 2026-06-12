--All the non-psychiatric (physical) main admissions that have icd codes and discharge notes, excluding the gey-zone physical admissions
CREATE OR REPLACE TABLE base_nonpsychiatric_admissions_with_discharge AS
SELECT DISTINCT
    a.subject_id,
    a.hadm_id,
    a.admittime
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

-- Case and Control and Mixxed subgroups
CREATE OR REPLACE TABLE base_admissions_mhc_subject_groups AS
WITH psychiatric_code_categories AS (
    SELECT icd_version, icd_code
    FROM psychiatric_icd_codes
),
same_admission_flags AS (
    SELECT DISTINCT
        b.subject_id,
        b.hadm_id,
        1 AS has_secondary_psychiatric_same_admission
    FROM base_nonpsychiatric_admissions_with_discharge b
    JOIN diagnoses_icd d
        ON b.subject_id = d.subject_id
       AND b.hadm_id = d.hadm_id
    JOIN psychiatric_code_categories p
        ON d.icd_version = p.icd_version
       AND d.icd_code = p.icd_code
    WHERE d.seq_num > 1
),
prior_history_flags AS (
    SELECT DISTINCT
        b.subject_id,
        b.hadm_id,
        1 AS has_prior_psychiatric_history
    FROM base_nonpsychiatric_admissions_with_discharge b
    WHERE EXISTS (
        SELECT 1
        FROM admissions a_prev
        JOIN diagnoses_icd d_prev
            ON a_prev.subject_id = d_prev.subject_id
           AND a_prev.hadm_id = d_prev.hadm_id
        JOIN psychiatric_code_categories p_prev
            ON d_prev.icd_version = p_prev.icd_version
           AND d_prev.icd_code = p_prev.icd_code
        WHERE a_prev.subject_id = b.subject_id
          AND a_prev.admittime < b.admittime
    )
),
admission_mhc AS (
    SELECT
        b.subject_id,
        b.hadm_id,
        b.admittime,
        COALESCE(s.has_secondary_psychiatric_same_admission, 0) AS has_secondary_psychiatric_same_admission,
        COALESCE(p.has_prior_psychiatric_history, 0) AS has_prior_psychiatric_history,
        CASE
            WHEN s.subject_id IS NOT NULL OR p.subject_id IS NOT NULL THEN 1
            ELSE 0
        END AS MHC
    FROM base_nonpsychiatric_admissions_with_discharge b
    LEFT JOIN same_admission_flags s
        ON b.subject_id = s.subject_id
       AND b.hadm_id = s.hadm_id
    LEFT JOIN prior_history_flags p
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
)
SELECT
    a.subject_id,
    a.hadm_id,
    a.admittime,
    a.MHC,
    a.has_secondary_psychiatric_same_admission,
    a.has_prior_psychiatric_history,
    CASE
        WHEN s.has_mhc0 = 1 AND s.has_mhc1 = 0 THEN 'only_MHC0'
        WHEN s.has_mhc0 = 0 AND s.has_mhc1 = 1 THEN 'only_MHC1'
        WHEN s.has_mhc0 = 1 AND s.has_mhc1 = 1 THEN 'MHC_0_to_1'
    END AS subject_group
FROM admission_mhc a
JOIN subject_groups s
    ON a.subject_id = s.subject_id;
