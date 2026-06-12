SELECT
    a.subject_id,
    a.hadm_id,
    a.admittime,
    a.dischtime,

    CASE
        WHEN b.hadm_id IS NOT NULL THEN 1
        ELSE 0
    END AS is_base_admission,

    b.subject_group,
    b.MHC,
    b.has_secondary_psychiatric_same_admission,
    b.has_prior_psychiatric_history,

    CASE
        WHEN b.subject_group = 'only_MHC1'
         AND b.has_secondary_psychiatric_same_admission = 1
        THEN 1
        ELSE 0
    END AS is_MHC1_sa,

    d1.icd_version AS primary_icd_version,
    d1.icd_code AS primary_icd_code,
    dd.long_title AS primary_long_title

FROM admissions a
LEFT JOIN base_admissions_mhc_subject_groups b
    ON a.subject_id = b.subject_id
   AND a.hadm_id = b.hadm_id
LEFT JOIN diagnoses_icd d1
    ON a.subject_id = d1.subject_id
   AND a.hadm_id = d1.hadm_id
   AND d1.seq_num = 1
LEFT JOIN d_icd_diagnoses dd
    ON d1.icd_version = dd.icd_version
   AND d1.icd_code = dd.icd_code
WHERE a.subject_id = 10000032
ORDER BY a.admittime;