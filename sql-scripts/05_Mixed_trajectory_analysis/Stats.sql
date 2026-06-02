-- Choose mixed subjects (patient level, we need all admissions/patient!)
CREATE OR REPLACE TABLE mixed_subjects AS
SELECT DISTINCT
    subject_id
FROM base_admissions_mhc_subject_groups
WHERE subject_group = 'MHC_0_to_1';
SELECT
    trajectory,
    COUNT(*) AS n_subjects
FROM mixed_group_trajectories
GROUP BY trajectory
ORDER BY n_subjects DESC, trajectory;

-- Their admissions
CREATE OR REPLACE TABLE mixed_subject_all_admissions AS
SELECT
    a.subject_id,
    a.hadm_id,
    a.admittime,
    a.dischtime
FROM admissions a
JOIN mixed_subjects m
    ON a.subject_id = m.subject_id;

--Mark which admissions are base admissions and what their MHC is
CREATE OR REPLACE TABLE mixed_subject_all_admissions_with_base_mhc AS
SELECT
    a.subject_id,
    a.hadm_id,
    a.admittime,
    a.dischtime,
    b.MHC,
    b.has_secondary_psychiatric_same_admission,
    b.has_prior_psychiatric_history,
    CASE
        WHEN b.hadm_id IS NOT NULL THEN 1
        ELSE 0
    END AS is_base_admission
FROM mixed_subject_all_admissions a
LEFT JOIN base_admissions_mhc_subject_groups b
    ON a.subject_id = b.subject_id
   AND a.hadm_id = b.hadm_id;

--Mark primary psychiatric admissions
CREATE OR REPLACE TABLE mixed_subject_all_admissions_with_flags AS
SELECT
    a.*,
    CASE
        WHEN EXISTS (
            SELECT 1
            FROM diagnoses_icd d1
            JOIN psychiatric_icd_codes p
                ON d1.icd_version = p.icd_version
               AND d1.icd_code = p.icd_code
            WHERE d1.subject_id = a.subject_id
              AND d1.hadm_id = a.hadm_id
              AND d1.seq_num = 1
        ) THEN 1
        ELSE 0
    END AS is_primary_psychiatric
FROM mixed_subject_all_admissions_with_base_mhc a;

--Assign trajectory states (0, 1, 4, 9)
CREATE OR REPLACE TABLE mixed_subject_admission_states AS
SELECT
    subject_id,
    hadm_id,
    admittime,
    dischtime,
    is_base_admission,
    MHC,
    is_primary_psychiatric,
    CASE
        WHEN is_base_admission = 1 AND MHC = 0 THEN 0
        WHEN is_base_admission = 1 AND MHC = 1 THEN 1
        WHEN is_primary_psychiatric = 1 THEN 4
        ELSE 9
    END AS admission_state
FROM mixed_subject_all_admissions_with_flags;

CREATE OR REPLACE TABLE mixed_group_trajectories AS
SELECT
    subject_id,
    COUNT(*) AS n_admissions,
    string_agg(CAST(admission_state AS VARCHAR), '-' ORDER BY admittime) AS trajectory
FROM mixed_subject_admission_states
GROUP BY subject_id
ORDER BY subject_id;

-- Assign trajectory states (0, 11, 12, 13, 4, 9)
CREATE OR REPLACE TABLE mixed_subject_admission_states AS
SELECT
    subject_id,
    hadm_id,
    admittime,
    dischtime,
    is_base_admission,
    MHC,
    is_primary_psychiatric,
    has_secondary_psychiatric_same_admission,
    has_prior_psychiatric_history,
    CASE
        WHEN is_base_admission = 1
         AND has_secondary_psychiatric_same_admission = 0
         AND has_prior_psychiatric_history = 0
            THEN 0
        WHEN is_base_admission = 1
         AND has_secondary_psychiatric_same_admission = 1
         AND has_prior_psychiatric_history = 0
            THEN 11
        WHEN is_base_admission = 1
         AND has_secondary_psychiatric_same_admission = 0
         AND has_prior_psychiatric_history = 1
            THEN 12
        WHEN is_base_admission = 1
         AND has_secondary_psychiatric_same_admission = 1
         AND has_prior_psychiatric_history = 1
            THEN 13
        WHEN is_primary_psychiatric = 1
            THEN 4
        ELSE 9
    END AS admission_state
FROM mixed_subject_all_admissions_with_flags;

CREATE OR REPLACE TABLE mixed_group_trajectories AS
SELECT
    subject_id,
    COUNT(*) AS n_admissions,
    string_agg(CAST(admission_state AS VARCHAR), '-' ORDER BY admittime) AS trajectory
FROM mixed_subject_admission_states
GROUP BY subject_id
ORDER BY subject_id;

SELECT
    trajectory,
    COUNT(*) AS n_subjects
FROM mixed_group_trajectories
GROUP BY trajectory
ORDER BY n_subjects DESC, trajectory;

SELECT
    admission_state,
    COUNT(DISTINCT hadm_id) AS n_admissions,
    COUNT(DISTINCT subject_id) AS n_subjects
FROM mixed_subject_admission_states
GROUP BY admission_state
ORDER BY admission_state;

SELECT
    trajectory,
    COUNT(*) AS n_subjects
FROM mixed_subject_trajectories
WHERE trajectory LIKE '%4%'
GROUP BY trajectory
ORDER BY n_subjects DESC, trajectory;