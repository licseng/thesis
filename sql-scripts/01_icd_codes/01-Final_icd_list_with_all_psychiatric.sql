-- 1) psychiatric diagnoses and suicide related, physical diagnosis
CREATE OR REPLACE TABLE psychiatric_icd_codes AS
SELECT DISTINCT
    icd_version,
    icd_code,
    long_title
FROM d_icd_diagnoses
WHERE
    (
        icd_version = 10
        AND icd_code LIKE 'F%'
    )
    OR
    (
        icd_version = 9
        AND (
            regexp_matches(icd_code, '^(29[0-9]|30[0-9]|31[0-9])')
            OR icd_code LIKE 'V11%'
        )
    )
    OR lower(long_title) LIKE '%suicid%'
    OR lower(long_title) LIKE '%self-harm%'
    OR lower(long_title) LIKE '%self harm%'
    OR lower(long_title) LIKE '%intentional self%';

