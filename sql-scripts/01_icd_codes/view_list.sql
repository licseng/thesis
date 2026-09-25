-- Canonical psychotic-illness definition, including clinical extensions and
-- the historical schizophrenia code used by the history-based MHH1 cohort.
SELECT *
FROM psychosis_icd_codes_extended
ORDER BY definition_tier, diagnostic_group, icd_version, icd_code;

SELECT
    definition_tier,
    diagnostic_group,
    COUNT(*) AS n_codes
FROM psychosis_icd_codes_extended
GROUP BY definition_tier, diagnostic_group
ORDER BY definition_tier, diagnostic_group;

SELECT *
FROM     psychiatric_icd_codes_psychotic
ORDER BY icd_version, icd_code;

SELECT *
FROM grey_zone_physical_icd_codes_candidates
ORDER BY icd_version, icd_code;
