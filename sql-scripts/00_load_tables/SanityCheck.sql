-- check if one discharge note / admission id - yes
SELECT
    n_notes_per_admission,
    COUNT(*) AS n_admissions
FROM (
    SELECT
        hadm_id,
        COUNT(*) AS n_notes_per_admission
    FROM discharge
    GROUP BY hadm_id
) t
GROUP BY n_notes_per_admission
ORDER BY n_notes_per_admission;