SELECT
    s.subject_id,
    s.hadm_id,
    s.admittime,
    s.admission_state,
    d1.icd_version AS primary_icd_version,
    d1.icd_code AS primary_icd_code,
    dd.long_title AS primary_long_title
FROM mixed_subject_admission_states s
LEFT JOIN diagnoses_icd d1
    ON s.subject_id = d1.subject_id
   AND s.hadm_id = d1.hadm_id
   AND d1.seq_num = 1
LEFT JOIN d_icd_diagnoses dd
    ON d1.icd_version = dd.icd_version
   AND d1.icd_code = dd.icd_code
WHERE s.subject_id = 11656371
ORDER BY s.admittime;