--unique patients in hosp -> 223,452
SELECT COUNT(DISTINCT subject_id) AS n_unique_patients
FROM admissions;

--unique patients in notes -> 145,914
SELECT COUNT(DISTINCT subject_id) AS n_subjects_with_discharge_note
FROM discharge;

--unique patients with notes and corresponding matched hosp admission -> 145,872
SELECT COUNT(DISTINCT a.subject_id) AS n_patients_with_discharge
FROM admissions a
JOIN discharge d
    ON a.subject_id = d.subject_id
   AND a.hadm_id = d.hadm_id;

--unique unmatched patients ->  48 (from which 6 patients are both unmatched and matched for different notes)
SELECT COUNT(DISTINCT d.subject_id) AS n_unmatched_subjects
FROM discharge d
LEFT JOIN admissions a
    ON a.subject_id = d.subject_id
   AND a.hadm_id = d.hadm_id
WHERE a.hadm_id IS NULL;

