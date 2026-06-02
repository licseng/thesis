SELECT *
FROM psychosis_icd_codes_extended
ORDER BY icd_version, icd_code;

SELECT *
FROM psychosis_icd_codes_restricted
ORDER BY icd_version, icd_code;

-- list of what has been removed from the extended list
SELECT
    e.icd_version,
    e.icd_code,
    e.long_title
FROM psychosis_icd_codes_extended e
LEFT JOIN psychosis_icd_codes_restricted r
    ON e.icd_version = r.icd_version
   AND e.icd_code = r.icd_code
WHERE r.icd_code IS NULL
ORDER BY e.icd_version, e.icd_code;

SELECT *
FROM     psychiatric_icd_codes_psychotic
ORDER BY icd_version, icd_code;

SELECT *
FROM grey_zone_physical_icd_codes_candidates
ORDER BY icd_version, icd_code;
