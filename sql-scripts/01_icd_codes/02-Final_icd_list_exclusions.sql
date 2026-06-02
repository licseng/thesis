-- Creates a grey-zone list of ambiguous physical ICD codes, including alcohol-related medical consequences and 
-- poisoning/intoxication-type diagnoses, to exclude from the clean physical base cohort.
CREATE OR REPLACE TABLE grey_zone_physical_icd_codes AS
SELECT
    icd_version,
    icd_code,
    long_title
FROM d_icd_diagnoses
WHERE
(
    icd_version = 9 AND (
        icd_code = '3575' OR
        icd_code = '4255' OR
        icd_code = '53530' OR
        icd_code = '53531' OR
        icd_code = '5710' OR
        icd_code = '5711' OR
        icd_code = '5712' OR
        icd_code = '5713'
    )
)
OR
(
    icd_version = 10 AND (
        icd_code = 'G621' OR
        icd_code = 'G721' OR
        icd_code = 'I426' OR
        icd_code = 'K292' OR
        icd_code = 'K2920' OR
        icd_code = 'K2921' OR
        icd_code = 'K70' OR
        icd_code = 'K700' OR
        icd_code = 'K701' OR
        icd_code = 'K7010' OR
        icd_code = 'K7011' OR
        icd_code = 'K702' OR
        icd_code = 'K703' OR
        icd_code = 'K7030' OR
        icd_code = 'K7031' OR
        icd_code = 'K704' OR
        icd_code = 'K7040' OR
        icd_code = 'K7041' OR
        icd_code = 'K709' OR
        icd_code = 'K860'
    )
)

UNION

SELECT DISTINCT
    d.icd_version,
    d.icd_code,
    d.long_title
FROM d_icd_diagnoses d
LEFT JOIN psychiatric_icd_codes p
    ON d.icd_version = p.icd_version
   AND d.icd_code = p.icd_code
LEFT JOIN psychiatric_icd_codes_suicide_self_harm s
    ON d.icd_version = s.icd_version
   AND d.icd_code = s.icd_code
WHERE p.icd_code IS NULL
  AND s.icd_code IS NULL
  AND (
        lower(d.long_title) LIKE '%poisoning%'
        OR lower(d.long_title) LIKE '%overdose%'
        OR lower(d.long_title) LIKE '%toxic effect%'
        OR lower(d.long_title) LIKE '%drug-induced%'
        OR lower(d.long_title) LIKE '%medication-induced%'
        OR lower(d.long_title) LIKE '%withdrawal%'
        OR lower(d.long_title) LIKE '%intoxication%'
  );
