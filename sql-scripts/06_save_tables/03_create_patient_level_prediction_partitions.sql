-- Patient-level train/validation/test-pool partitions for prediction-model work.
--
-- Intended use:
--   Run this in DBeaver against the local DuckDB database.
--
-- Purpose:
--   1. Start from the broad eligible MIMIC population with discharge notes.
--   2. Randomly sample patients into train, targeting ~80,000 admissions.
--   3. From the remaining patients, randomly sample validation, targeting
--      ~10,000 admissions.
--   4. Leave remaining patients as test_pool.
--   5. From test_pool, randomly flag a general test sample targeting ~20,000
--      admissions. This is a test sample flag, not a new partition.
--   6. Keep all admissions for the same subject_id in the same partition/test
--      sample flag.
--   7. Report how many MHH1/MHC0 patients/admissions land in each partition
--      and how much the general test sample overlaps with MHH1/MHC0.
--
-- Important:
--   This does not create prediction labels or model inputs yet.
--   This does not alter existing MHH1/MHC0/matched-cohort tables.
--
-- Notes:
--   DuckDB random seeding can be session/version dependent. For reproducibility,
--   this script uses deterministic hash ordering based on subject_id plus fixed
--   seed strings. Re-running the script with the same eligible population gives
--   the same partitions.


-- ---------------------------------------------------------------------------
-- 0. Settings
-- ---------------------------------------------------------------------------

CREATE OR REPLACE TEMP TABLE prediction_partition_settings AS
SELECT
    80000::BIGINT AS target_train_admissions,
    10000::BIGINT AS target_validation_admissions,
    20000::BIGINT AS target_general_test_admissions,
    'prediction_partition_seed_2026_09_03' AS seed_string;


-- ---------------------------------------------------------------------------
-- 1. Eligible admission population
-- ---------------------------------------------------------------------------
-- If you already have a stricter "eligible discharge-note admission" table,
-- replace this table definition with a SELECT from that table.
--
-- The current default is broad: one row per hospital admission with a discharge
-- note and an admissions-table match.

CREATE OR REPLACE TABLE eligible_prediction_admissions AS
SELECT DISTINCT
    adm.subject_id,
    adm.hadm_id,
    adm.admittime,
    adm.dischtime,
    adm.deathtime,
    adm.admission_type,
    adm.admission_location,
    adm.discharge_location,
    adm.insurance,
    adm.language,
    adm.race,
    adm.marital_status,
    adm.edregtime,
    adm.edouttime,
    adm.hospital_expire_flag,
    pat.gender,
    pat.anchor_age,
    pat.anchor_year,
    pat.anchor_year_group,
    pat.dod,
    dis.note_id,
    dis.charttime AS discharge_note_charttime,
    dis.storetime AS discharge_note_storetime
FROM admissions adm
JOIN discharge dis
    ON adm.subject_id = dis.subject_id
   AND adm.hadm_id = dis.hadm_id
LEFT JOIN patients pat
    ON adm.subject_id = pat.subject_id
WHERE adm.subject_id IS NOT NULL
  AND adm.hadm_id IS NOT NULL;


-- ---------------------------------------------------------------------------
-- 2. Patient-level group flags for later reporting
-- ---------------------------------------------------------------------------
-- These flags are used only for summaries. They do not affect sampling.
--
-- is_mhh1_psychotic_* uses the existing psychosis-history cohort table.
-- is_mhc0_* uses the existing no-mental-health-context/history cohort table.
-- is_matched_* uses matched_cohort if present in the database.

CREATE OR REPLACE TABLE eligible_prediction_admissions_with_group_flags AS
WITH
mhh1_admissions AS (
    SELECT DISTINCT
        subject_id,
        hadm_id
    FROM export_MHH_psychotic
),

mhc0_admissions AS (
    SELECT DISTINCT
        subject_id,
        hadm_id
    FROM export_only_MHC0
),

matched_admissions AS (
    SELECT DISTINCT
        cohort,
        subject_id,
        hadm_id
    FROM matched_cohort
),

admission_flags AS (
    SELECT
        e.*,

        CASE WHEN mhh1.hadm_id IS NOT NULL THEN 1 ELSE 0 END
            AS is_mhh1_psychotic_admission,
        CASE WHEN mhc0.hadm_id IS NOT NULL THEN 1 ELSE 0 END
            AS is_mhc0_admission,
        CASE
            WHEN matched.cohort = 'MHH1_psychotic' THEN 1 ELSE 0
        END AS is_matched_mhh1_psychotic_admission,
        CASE
            WHEN matched.cohort = 'MHC0' THEN 1 ELSE 0
        END AS is_matched_mhc0_admission

    FROM eligible_prediction_admissions e

    LEFT JOIN mhh1_admissions mhh1
        ON e.subject_id = mhh1.subject_id
       AND e.hadm_id = mhh1.hadm_id

    LEFT JOIN mhc0_admissions mhc0
        ON e.subject_id = mhc0.subject_id
       AND e.hadm_id = mhc0.hadm_id

    LEFT JOIN matched_admissions matched
        ON e.subject_id = matched.subject_id
       AND e.hadm_id = matched.hadm_id
)

SELECT
    admission_flags.*,

    MAX(is_mhh1_psychotic_admission) OVER (PARTITION BY subject_id)
        AS is_mhh1_psychotic_subject,
    MAX(is_mhc0_admission) OVER (PARTITION BY subject_id)
        AS is_mhc0_subject,
    MAX(is_matched_mhh1_psychotic_admission) OVER (PARTITION BY subject_id)
        AS is_matched_mhh1_psychotic_subject,
    MAX(is_matched_mhc0_admission) OVER (PARTITION BY subject_id)
        AS is_matched_mhc0_subject

FROM admission_flags;


-- ---------------------------------------------------------------------------
-- 3. Deterministic patient-level partitioning
-- ---------------------------------------------------------------------------
-- Patients are sampled into train first, then removed before validation.
-- The test_pool is every remaining patient. A patient-level general test
-- sample is then selected from test_pool without removing MHH1/MHC0 patients
-- from the later fairness-evaluation view.

CREATE OR REPLACE TABLE eligible_prediction_subjects AS
SELECT
    subject_id,
    COUNT(DISTINCT hadm_id) AS n_eligible_admissions,
    MAX(is_mhh1_psychotic_subject) AS is_mhh1_psychotic_subject,
    MAX(is_mhc0_subject) AS is_mhc0_subject,
    MAX(is_matched_mhh1_psychotic_subject) AS is_matched_mhh1_psychotic_subject,
    MAX(is_matched_mhc0_subject) AS is_matched_mhc0_subject
FROM eligible_prediction_admissions_with_group_flags
GROUP BY subject_id;


CREATE OR REPLACE TEMP TABLE train_subject_candidates AS
WITH randomized AS (
    SELECT
        s.*,
        hash(CAST(s.subject_id AS VARCHAR) || '|train|' || p.seed_string)
            AS train_random_key
    FROM eligible_prediction_subjects s
    CROSS JOIN prediction_partition_settings p
),

ordered AS (
    SELECT
        *,
        SUM(n_eligible_admissions) OVER (
            ORDER BY train_random_key, subject_id
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS cumulative_admissions_before_subject
    FROM randomized
)

SELECT *
FROM ordered
CROSS JOIN prediction_partition_settings p
WHERE COALESCE(cumulative_admissions_before_subject, 0)
    < p.target_train_admissions;


CREATE OR REPLACE TEMP TABLE validation_subject_candidates AS
WITH remaining_subjects AS (
    SELECT s.*
    FROM eligible_prediction_subjects s
    LEFT JOIN train_subject_candidates train
        ON s.subject_id = train.subject_id
    WHERE train.subject_id IS NULL
),

randomized AS (
    SELECT
        s.*,
        hash(CAST(s.subject_id AS VARCHAR) || '|validation|' || p.seed_string)
            AS validation_random_key
    FROM remaining_subjects s
    CROSS JOIN prediction_partition_settings p
),

ordered AS (
    SELECT
        *,
        SUM(n_eligible_admissions) OVER (
            ORDER BY validation_random_key, subject_id
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS cumulative_admissions_before_subject
    FROM randomized
)

SELECT *
FROM ordered
CROSS JOIN prediction_partition_settings p
WHERE COALESCE(cumulative_admissions_before_subject, 0)
    < p.target_validation_admissions;


CREATE OR REPLACE TABLE patient_partition AS
SELECT
    s.subject_id,
    CASE
        WHEN train.subject_id IS NOT NULL THEN 'train'
        WHEN validation.subject_id IS NOT NULL THEN 'validation'
        ELSE 'test_pool'
    END AS partition
FROM eligible_prediction_subjects s
LEFT JOIN train_subject_candidates train
    ON s.subject_id = train.subject_id
LEFT JOIN validation_subject_candidates validation
    ON s.subject_id = validation.subject_id;


CREATE OR REPLACE TEMP TABLE general_test_subject_candidates AS
WITH test_pool_subjects AS (
    SELECT s.*
    FROM eligible_prediction_subjects s
    JOIN patient_partition p
        ON s.subject_id = p.subject_id
    WHERE p.partition = 'test_pool'
),

randomized AS (
    SELECT
        s.*,
        hash(CAST(s.subject_id AS VARCHAR) || '|general_test|' || p.seed_string)
            AS general_test_random_key
    FROM test_pool_subjects s
    CROSS JOIN prediction_partition_settings p
),

ordered AS (
    SELECT
        *,
        SUM(n_eligible_admissions) OVER (
            ORDER BY general_test_random_key, subject_id
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS cumulative_admissions_before_subject
    FROM randomized
)

SELECT *
FROM ordered
CROSS JOIN prediction_partition_settings p
WHERE COALESCE(cumulative_admissions_before_subject, 0)
    < p.target_general_test_admissions;


CREATE OR REPLACE TABLE patient_partition_with_test_flags AS
SELECT
    p.subject_id,
    p.partition,
    CASE WHEN general_test.subject_id IS NOT NULL THEN 1 ELSE 0 END
        AS selected_for_general_test
FROM patient_partition p
LEFT JOIN general_test_subject_candidates general_test
    ON p.subject_id = general_test.subject_id;


CREATE OR REPLACE TABLE eligible_prediction_admissions_with_partition AS
SELECT
    e.*,
    p.partition,
    p.selected_for_general_test
FROM eligible_prediction_admissions_with_group_flags e
JOIN patient_partition_with_test_flags p
    ON e.subject_id = p.subject_id;


-- ---------------------------------------------------------------------------
-- 4. Partition-size summaries
-- ---------------------------------------------------------------------------

CREATE OR REPLACE TABLE patient_partition_summary AS
SELECT
    partition,
    COUNT(DISTINCT subject_id) AS n_subjects,
    COUNT(DISTINCT hadm_id) AS n_admissions
FROM eligible_prediction_admissions_with_partition
GROUP BY partition
ORDER BY
    CASE partition
        WHEN 'train' THEN 1
        WHEN 'validation' THEN 2
        WHEN 'test_pool' THEN 3
        ELSE 4
    END;


CREATE OR REPLACE TABLE patient_partition_group_summary AS
WITH group_long AS (
    SELECT
        partition,
        'MHH1_psychotic_full_cohort' AS group_name,
        subject_id,
        hadm_id
    FROM eligible_prediction_admissions_with_partition
    WHERE is_mhh1_psychotic_subject = 1
      AND is_mhh1_psychotic_admission = 1

    UNION ALL

    SELECT
        partition,
        'MHC0_full_cohort' AS group_name,
        subject_id,
        hadm_id
    FROM eligible_prediction_admissions_with_partition
    WHERE is_mhc0_subject = 1
      AND is_mhc0_admission = 1

    UNION ALL

    SELECT
        partition,
        'MHH1_psychotic_matched_cohort' AS group_name,
        subject_id,
        hadm_id
    FROM eligible_prediction_admissions_with_partition
    WHERE is_matched_mhh1_psychotic_subject = 1
      AND is_matched_mhh1_psychotic_admission = 1

    UNION ALL

    SELECT
        partition,
        'MHC0_matched_cohort' AS group_name,
        subject_id,
        hadm_id
    FROM eligible_prediction_admissions_with_partition
    WHERE is_matched_mhc0_subject = 1
      AND is_matched_mhc0_admission = 1
)

SELECT
    group_name,
    partition,
    COUNT(DISTINCT subject_id) AS n_subjects,
    COUNT(DISTINCT hadm_id) AS n_admissions
FROM group_long
GROUP BY group_name, partition
ORDER BY
    group_name,
    CASE partition
        WHEN 'train' THEN 1
        WHEN 'validation' THEN 2
        WHEN 'test_pool' THEN 3
        ELSE 4
    END;


CREATE OR REPLACE TABLE patient_partition_unseen_fairness_pool_summary AS
SELECT
    group_name,
    n_subjects AS n_unseen_subjects_in_test_pool,
    n_admissions AS n_unseen_admissions_in_test_pool
FROM patient_partition_group_summary
WHERE partition = 'test_pool'
ORDER BY group_name;


CREATE OR REPLACE TABLE patient_partition_general_test_summary AS
SELECT
    selected_for_general_test,
    COUNT(DISTINCT subject_id) AS n_subjects,
    COUNT(DISTINCT hadm_id) AS n_admissions
FROM eligible_prediction_admissions_with_partition
WHERE partition = 'test_pool'
GROUP BY selected_for_general_test
ORDER BY selected_for_general_test DESC;


CREATE OR REPLACE TABLE patient_partition_general_test_group_overlap AS
WITH group_long AS (
    SELECT
        'MHH1_psychotic_full_cohort' AS group_name,
        subject_id,
        hadm_id
    FROM eligible_prediction_admissions_with_partition
    WHERE partition = 'test_pool'
      AND selected_for_general_test = 1
      AND is_mhh1_psychotic_subject = 1
      AND is_mhh1_psychotic_admission = 1

    UNION ALL

    SELECT
        'MHC0_full_cohort' AS group_name,
        subject_id,
        hadm_id
    FROM eligible_prediction_admissions_with_partition
    WHERE partition = 'test_pool'
      AND selected_for_general_test = 1
      AND is_mhc0_subject = 1
      AND is_mhc0_admission = 1

    UNION ALL

    SELECT
        'MHH1_psychotic_matched_cohort' AS group_name,
        subject_id,
        hadm_id
    FROM eligible_prediction_admissions_with_partition
    WHERE partition = 'test_pool'
      AND selected_for_general_test = 1
      AND is_matched_mhh1_psychotic_subject = 1
      AND is_matched_mhh1_psychotic_admission = 1

    UNION ALL

    SELECT
        'MHC0_matched_cohort' AS group_name,
        subject_id,
        hadm_id
    FROM eligible_prediction_admissions_with_partition
    WHERE partition = 'test_pool'
      AND selected_for_general_test = 1
      AND is_matched_mhc0_subject = 1
      AND is_matched_mhc0_admission = 1
)

SELECT
    group_name,
    COUNT(DISTINCT subject_id) AS n_subjects_in_general_test,
    COUNT(DISTINCT hadm_id) AS n_admissions_in_general_test
FROM group_long
GROUP BY group_name
ORDER BY group_name;


-- ---------------------------------------------------------------------------
-- 5. Display results in DBeaver
-- ---------------------------------------------------------------------------

SELECT *
FROM patient_partition_summary;

SELECT *
FROM patient_partition_group_summary;

SELECT *
FROM patient_partition_unseen_fairness_pool_summary;

SELECT *
FROM patient_partition_general_test_summary;

SELECT *
FROM patient_partition_general_test_group_overlap;

-- QC: no subject should appear in more than one partition.
SELECT
    COUNT(*) AS n_subjects_with_multiple_partitions
FROM (
    SELECT
        subject_id,
        COUNT(DISTINCT partition) AS n_partitions
    FROM patient_partition
    GROUP BY subject_id
    HAVING COUNT(DISTINCT partition) > 1
) duplicated_subjects;

-- QC: the general test sample should only come from test_pool.
SELECT
    COUNT(*) AS n_non_test_pool_subjects_selected_for_general_test
FROM patient_partition_with_test_flags
WHERE selected_for_general_test = 1
  AND partition <> 'test_pool';
