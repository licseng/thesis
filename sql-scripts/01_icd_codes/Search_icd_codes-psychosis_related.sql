-- Checks for psychosis-related ICD titles that are not yet included in the current psychotic subgroup definition. Helper for the extended psychosis icd list.
SELECT
    icd_version,
    icd_code,
    long_title
FROM d_icd_diagnoses
WHERE
    (
        lower(long_title) LIKE '%psycho%'
        OR lower(long_title) LIKE '%schizo%'
        OR lower(long_title) LIKE '%delusion%'

    )
    AND NOT (
        (icd_version = 10 AND (
            icd_code LIKE 'F20%' OR
            icd_code LIKE 'F22%' OR
            icd_code LIKE 'F23%' OR
            icd_code LIKE 'F24%' OR
            icd_code LIKE 'F25%' OR
            icd_code LIKE 'F28%' OR
            icd_code LIKE 'F29%' OR
            icd_code LIKE 'F060%' OR
            icd_code LIKE 'F061%' OR
            icd_code LIKE 'F062%' OR
            icd_code LIKE 'F531%' OR
            icd_code LIKE 'F10%5%' OR
            icd_code LIKE 'F11%5%' OR
            icd_code LIKE 'F12%5%' OR
            icd_code LIKE 'F13%5%' OR
            icd_code LIKE 'F14%5%' OR
            icd_code LIKE 'F15%5%' OR
            icd_code LIKE 'F16%5%' OR
            icd_code LIKE 'F17%5%' OR
            icd_code LIKE 'F18%5%' OR
            icd_code LIKE 'F19%5%'
        ))
        OR
        (icd_version = 9 AND (
            icd_code LIKE '295%' OR
            icd_code = '2971' OR
            icd_code = '2973' OR
            icd_code = '2988' OR
            icd_code = '2989' OR
            icd_code = '29381' OR
            icd_code = '29382' OR
            icd_code = '29211' OR
            icd_code = '29212' OR
            icd_code LIKE '6484%'
        ))
    )
ORDER BY icd_version, icd_code;