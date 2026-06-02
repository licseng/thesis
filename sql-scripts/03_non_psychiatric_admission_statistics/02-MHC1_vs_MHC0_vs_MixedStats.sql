-- Describes the pure MHC0 and MHC1 subject groups in terms of cohort size, age, sex, admissions per subject, and top primary diagnoses.
SELECT
    subject_group,
    MHC,
    COUNT(DISTINCT hadm_id) AS n_admissions,
    COUNT(DISTINCT subject_id) AS n_subjects
FROM base_admissions_mhc_subject_groups
GROUP BY subject_group, MHC
ORDER BY subject_group, MHC;

WITH subject_groups AS (
    SELECT DISTINCT
        subject_group,
        subject_id
    FROM base_admissions_mhc_subject_groups
    WHERE subject_group IN ('only_MHC0', 'only_MHC1')
)
SELECT
    s.subject_group,
    COUNT(*) AS n_subjects,
    ROUND(AVG(p.anchor_age), 2) AS mean_anchor_age,
    MIN(p.anchor_age) AS min_anchor_age,
    MAX(p.anchor_age) AS max_anchor_age
FROM subject_groups s
JOIN patients p
    ON s.subject_id = p.subject_id
GROUP BY s.subject_group
ORDER BY s.subject_group;

WITH subject_groups AS (
    SELECT DISTINCT
        subject_group,
        subject_id
    FROM base_admissions_mhc_subject_groups
    WHERE subject_group IN ('only_MHC0', 'only_MHC1')
)
SELECT
    s.subject_group,
    p.gender,
    COUNT(*) AS n_subjects
FROM subject_groups s
JOIN patients p
    ON s.subject_id = p.subject_id
GROUP BY s.subject_group, p.gender
ORDER BY s.subject_group, p.gender;

WITH subject_admission_counts AS (
    SELECT
        subject_group,
        subject_id,
        COUNT(DISTINCT hadm_id) AS n_admissions
    FROM base_admissions_mhc_subject_groups
    WHERE subject_group IN ('only_MHC0', 'only_MHC1')
    GROUP BY subject_group, subject_id
)
SELECT
    subject_group,
    COUNT(*) AS n_subjects,
    ROUND(AVG(n_admissions), 3) AS mean_admissions_per_subject,
    MIN(n_admissions) AS min_admissions_per_subject,
    MAX(n_admissions) AS max_admissions_per_subject
FROM subject_admission_counts
GROUP BY subject_group
ORDER BY subject_group;

WITH dx_counts AS (
    SELECT
        g.subject_group,
        d1.icd_version,
        d1.icd_code,
        dd.long_title,
        COUNT(DISTINCT g.hadm_id) AS n_admissions
    FROM base_admissions_mhc_subject_groups g
    JOIN diagnoses_icd d1
        ON g.subject_id = d1.subject_id
       AND g.hadm_id = d1.hadm_id
       AND d1.seq_num = 1
    LEFT JOIN d_icd_diagnoses dd
        ON d1.icd_version = dd.icd_version
       AND d1.icd_code = dd.icd_code
    WHERE g.subject_group IN ('only_MHC0', 'only_MHC1')
    GROUP BY
        g.subject_group,
        d1.icd_version,
        d1.icd_code,
        dd.long_title
),
ranked AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY subject_group
            ORDER BY n_admissions DESC
        ) AS rn
    FROM dx_counts
)
SELECT *
FROM ranked
WHERE rn <= 15
ORDER BY subject_group, rn;


