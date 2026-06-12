-- Creates a list of ambiguous physical/non-psychiatric ICD codes that may still reflect psychiatric, substance-related, self-harm, or mental-status-related presentations.
CREATE OR REPLACE TABLE ambiguous_physical_icd_codes AS
SELECT DISTINCT
    icd_version,
    icd_code,
    long_title
FROM d_icd_diagnoses
WHERE
    lower(long_title) LIKE '%suicid%'
    OR lower(long_title) LIKE '%self-harm%'
    OR lower(long_title) LIKE '%self harm%'
    OR lower(long_title) LIKE '%intentional self%'
    OR lower(long_title) LIKE '%alcohol%'
    OR lower(long_title) LIKE '%overdose%'
    OR lower(long_title) LIKE '%toxic effect%'
    OR lower(long_title) LIKE '%altered mental status%'
    OR lower(long_title) LIKE '%confusion%'
    OR lower(long_title) LIKE '%delirium%'
ORDER BY icd_version, icd_code;