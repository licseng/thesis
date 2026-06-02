-- Sanity-checks the exported case, control, and mixed-group tables by verifying cohort size, note availability, 
-- psychiatric-code presence, MHC values, and component-flag distributions.
SELECT
    'export_only_MHC1_same_admission' AS table_name,
    COUNT(DISTINCT hadm_id) AS n_admissions,
    COUNT(DISTINCT subject_id) AS n_subjects
FROM export_only_MHC1_same_admission

UNION ALL

SELECT
    'export_only_MHC0' AS table_name,
    COUNT(DISTINCT hadm_id) AS n_admissions,
    COUNT(DISTINCT subject_id) AS n_subjects
FROM export_only_MHC0

UNION ALL

SELECT
    'export_mixed_group' AS table_name,
    COUNT(DISTINCT hadm_id) AS n_admissions,
    COUNT(DISTINCT subject_id) AS n_subjects
FROM export_mixed_group;

SELECT
    'export_only_MHC1_same_admission' AS table_name,
    COUNT(DISTINCT hadm_id) AS n_admissions,
    COUNT(DISTINCT CASE WHEN text IS NOT NULL AND length(trim(text)) > 0 THEN hadm_id END) AS n_with_note_text
FROM export_only_MHC1_same_admission

UNION ALL

SELECT
    'export_only_MHC0' AS table_name,
    COUNT(DISTINCT hadm_id) AS n_admissions,
    COUNT(DISTINCT CASE WHEN text IS NOT NULL AND length(trim(text)) > 0 THEN hadm_id END) AS n_with_note_text
FROM export_only_MHC0

UNION ALL

SELECT
    'export_mixed_group' AS table_name,
    COUNT(DISTINCT hadm_id) AS n_admissions,
    COUNT(DISTINCT CASE WHEN text IS NOT NULL AND length(trim(text)) > 0 THEN hadm_id END) AS n_with_note_text
FROM export_mixed_group;

SELECT
    'export_only_MHC1_same_admission' AS table_name,
    COUNT(DISTINCT e.hadm_id) AS n_admissions,
    COUNT(DISTINCT CASE WHEN EXISTS (
        SELECT 1
        FROM diagnoses_icd d
        JOIN psychiatric_icd_codes p
            ON d.icd_version = p.icd_version
           AND d.icd_code = p.icd_code
        WHERE d.subject_id = e.subject_id
          AND d.hadm_id = e.hadm_id
          AND d.seq_num > 1
    ) THEN e.hadm_id END) AS n_with_secondary_psychiatric_code
FROM export_only_MHC1_same_admission e

UNION ALL

SELECT
    'export_only_MHC0' AS table_name,
    COUNT(DISTINCT e.hadm_id) AS n_admissions,
    COUNT(DISTINCT CASE WHEN EXISTS (
        SELECT 1
        FROM diagnoses_icd d
        JOIN psychiatric_icd_codes p
            ON d.icd_version = p.icd_version
           AND d.icd_code = p.icd_code
        WHERE d.subject_id = e.subject_id
          AND d.hadm_id = e.hadm_id
          AND d.seq_num > 1
    ) THEN e.hadm_id END) AS n_with_secondary_psychiatric_code
FROM export_only_MHC0 e

UNION ALL

SELECT
    'export_mixed_group' AS table_name,
    COUNT(DISTINCT e.hadm_id) AS n_admissions,
    COUNT(DISTINCT CASE WHEN EXISTS (
        SELECT 1
        FROM diagnoses_icd d
        JOIN psychiatric_icd_codes p
            ON d.icd_version = p.icd_version
           AND d.icd_code = p.icd_code
        WHERE d.subject_id = e.subject_id
          AND d.hadm_id = e.hadm_id
          AND d.seq_num > 1
    ) THEN e.hadm_id END) AS n_with_secondary_psychiatric_code
FROM export_mixed_group e;

SELECT
    'export_only_MHC1_same_admission' AS table_name,
    MHC,
    COUNT(DISTINCT hadm_id) AS n_admissions
FROM export_only_MHC1_same_admission
GROUP BY MHC

UNION ALL

SELECT
    'export_only_MHC0' AS table_name,
    MHC,
    COUNT(DISTINCT hadm_id) AS n_admissions
FROM export_only_MHC0
GROUP BY MHC

UNION ALL

SELECT
    'export_mixed_group' AS table_name,
    MHC,
    COUNT(DISTINCT hadm_id) AS n_admissions
FROM export_mixed_group
GROUP BY MHC
ORDER BY table_name, MHC;

SELECT
    'export_only_MHC1_same_admission' AS table_name,
    has_secondary_psychiatric_same_admission,
    has_prior_psychiatric_history,
    COUNT(DISTINCT hadm_id) AS n_admissions
FROM export_only_MHC1_same_admission
GROUP BY 2,3

UNION ALL

SELECT
    'export_only_MHC0' AS table_name,
    has_secondary_psychiatric_same_admission,
    has_prior_psychiatric_history,
    COUNT(DISTINCT hadm_id) AS n_admissions
FROM export_only_MHC0
GROUP BY 2,3

UNION ALL

SELECT
    'export_mixed_group' AS table_name,
    has_secondary_psychiatric_same_admission,
    has_prior_psychiatric_history,
    COUNT(DISTINCT hadm_id) AS n_admissions
FROM export_mixed_group
GROUP BY 2,3
ORDER BY table_name, has_secondary_psychiatric_same_admission, has_prior_psychiatric_history;