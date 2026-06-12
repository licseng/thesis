SELECT
    'MHH_psychotic' AS group_name,
    COUNT(*) AS n_rows,
    COUNT(DISTINCT hadm_id) AS n_admissions,
    MIN(age_at_admission) AS min_age,
    MAX(age_at_admission) AS max_age,
    SUM(CASE WHEN sex NOT IN ('F', 'M') THEN 1 ELSE 0 END) AS n_unexpected_sex
FROM export_MHH_psychotic

UNION ALL

SELECT
    'MHC0' AS group_name,
    COUNT(*) AS n_rows,
    COUNT(DISTINCT hadm_id) AS n_admissions,
    MIN(age_at_admission) AS min_age,
    MAX(age_at_admission) AS max_age,
    SUM(CASE WHEN sex NOT IN ('F', 'M') THEN 1 ELSE 0 END) AS n_unexpected_sex
FROM export_only_MHC0;