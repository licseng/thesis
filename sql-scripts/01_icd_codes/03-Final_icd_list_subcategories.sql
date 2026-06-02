-- Splits the broad psychiatric ICD code list into diagnostic subgroup tables and reports the number of codes in each subgroup.
CREATE OR REPLACE TABLE psychiatric_icd_codes_psychotic AS
SELECT DISTINCT
    icd_version,
    icd_code,
    long_title
FROM psychiatric_icd_codes
WHERE
    (icd_version = 10 AND (
        icd_code LIKE 'F20%' OR
        icd_code LIKE 'F21%' OR
        icd_code LIKE 'F22%' OR
        icd_code LIKE 'F23%' OR
        icd_code LIKE 'F24%' OR
        icd_code LIKE 'F25%' OR
        icd_code LIKE 'F28%' OR
        icd_code LIKE 'F29%'
    ))
    OR
    (icd_version = 9 AND (
        icd_code LIKE '295%' OR
        icd_code LIKE '297%' OR
        icd_code LIKE '298%' OR
        icd_code = 'V110'
    ));

CREATE OR REPLACE TABLE psychiatric_icd_codes_substance_related AS
SELECT DISTINCT
    icd_version,
    icd_code,
    long_title
FROM psychiatric_icd_codes
WHERE
    (icd_version = 10 AND (
        icd_code LIKE 'F10%' OR
        icd_code LIKE 'F11%' OR
        icd_code LIKE 'F12%' OR
        icd_code LIKE 'F13%' OR
        icd_code LIKE 'F14%' OR
        icd_code LIKE 'F15%' OR
        icd_code LIKE 'F16%' OR
        icd_code LIKE 'F17%' OR
        icd_code LIKE 'F18%' OR
        icd_code LIKE 'F19%'
    ))
    OR
    (icd_version = 9 AND (
        icd_code LIKE '291%' OR
        icd_code LIKE '292%' OR
        icd_code LIKE '303%' OR
        icd_code LIKE '304%' OR
        icd_code LIKE '305%' OR
        icd_code = 'V113'

    ));

CREATE OR REPLACE TABLE psychiatric_icd_codes_internalizing AS
SELECT DISTINCT
    icd_version,
    icd_code,
    long_title
FROM psychiatric_icd_codes
WHERE
    /* ICD-10 mood/anxiety/stress/somatic/eating */
    (icd_version = 10 AND (
        icd_code LIKE 'F30%' OR
        icd_code LIKE 'F31%' OR
        icd_code LIKE 'F32%' OR
        icd_code LIKE 'F33%' OR
        icd_code LIKE 'F34%' OR
        icd_code LIKE 'F38%' OR
        icd_code LIKE 'F39%' OR
        icd_code LIKE 'F40%' OR
        icd_code LIKE 'F41%' OR
        icd_code LIKE 'F42%' OR
        icd_code LIKE 'F43%' OR
        icd_code LIKE 'F44%' OR
        icd_code LIKE 'F45%' OR
        icd_code LIKE 'F48%' OR
        icd_code LIKE 'F50%' 
    ))
    OR
    /* ICD-9 mood/anxiety/stress/eating */
	(icd_version = 9 AND (
	    icd_code LIKE '296%' OR
	    icd_code LIKE '300%' OR
	    icd_code LIKE '306%' OR
	    icd_code LIKE '308%' OR
	    icd_code LIKE '309%' OR
	    icd_code = '311' OR
	    icd_code LIKE '3071%' OR
	    icd_code = '3130' OR
		icd_code = '3131' OR
		icd_code LIKE '3132%' OR
		icd_code = 'V111' OR
        icd_code = 'V112' OR
        icd_code = 'V114'

));

CREATE OR REPLACE TABLE psychiatric_icd_codes_personality_behavioral AS
SELECT DISTINCT
    icd_version,
    icd_code,
    long_title
FROM psychiatric_icd_codes
WHERE
    /* ICD-10 personality + adult behavioral syndromes except eating */
    (icd_version = 10 AND (
        icd_code LIKE 'F60%' OR
        icd_code LIKE 'F61%' OR
        icd_code LIKE 'F62%' OR
        icd_code LIKE 'F63%' OR
        icd_code LIKE 'F64%' OR
        icd_code LIKE 'F65%' OR
        icd_code LIKE 'F66%' OR
        icd_code LIKE 'F68%' OR
        icd_code LIKE 'F69%' OR
        icd_code LIKE 'F51%' OR
        icd_code LIKE 'F52%' OR
        icd_code LIKE 'F53%' OR
        icd_code LIKE 'F54%' OR
        icd_code LIKE 'F55%' OR
        icd_code LIKE 'F59%'
    ))
    OR
    /* ICD-9 personality + sexual/behavioral syndromes */
    (icd_version = 9 AND (
        icd_code LIKE '301%' OR
        icd_code LIKE '302%' OR
        icd_code LIKE '3070%' OR
        icd_code LIKE '3073%' OR
        icd_code LIKE '3074%' OR
        icd_code LIKE '3075%' OR
        icd_code LIKE '3076%' OR
        icd_code LIKE '3077%' OR
        icd_code LIKE '3078%' OR
        icd_code LIKE '3079%' OR
        icd_code LIKE '312%' OR
		icd_code LIKE '3133%' OR
		icd_code LIKE '3138%' OR
		icd_code = '3139'
    ));

CREATE OR REPLACE TABLE psychiatric_icd_codes_neurodevelopmental AS
SELECT DISTINCT
    icd_version,
    icd_code,
    long_title
FROM psychiatric_icd_codes
WHERE
    /* ICD-10 neurodevelopmental / childhood-onset */
    (icd_version = 10 AND (
        icd_code LIKE 'F70%' OR
        icd_code LIKE 'F71%' OR
        icd_code LIKE 'F72%' OR
        icd_code LIKE 'F73%' OR
        icd_code LIKE 'F78%' OR
        icd_code LIKE 'F79%' OR
        icd_code LIKE 'F80%' OR
        icd_code LIKE 'F81%' OR
        icd_code LIKE 'F82%' OR
        icd_code LIKE 'F83%' OR
        icd_code LIKE 'F84%' OR
        icd_code LIKE 'F88%' OR
        icd_code LIKE 'F89%' OR
        icd_code LIKE 'F90%' OR
        icd_code LIKE 'F91%' OR
        icd_code LIKE 'F92%' OR
        icd_code LIKE 'F93%' OR
        icd_code LIKE 'F94%' OR
        icd_code LIKE 'F95%' OR
        icd_code LIKE 'F98%'
    ))
    OR
    /* ICD-9 ADHD / autism / tic / developmental / intellectual disability */
    (icd_version = 9 AND (
        icd_code LIKE '299%' OR
        icd_code LIKE '3072%' OR
        icd_code LIKE '314%' OR
        icd_code LIKE '315%' OR
        icd_code LIKE '317%' OR
        icd_code LIKE '318%' OR
        icd_code LIKE '319%'
    ));

CREATE OR REPLACE TABLE psychiatric_icd_codes_neurocognitive AS
SELECT DISTINCT
    icd_version,
    icd_code,
    long_title
FROM psychiatric_icd_codes
WHERE
    (icd_version = 10 AND (
        icd_code LIKE 'F01%' OR
        icd_code LIKE 'F02%' OR
        icd_code LIKE 'F03%'
    ))
    OR
    (icd_version = 9 AND (
        icd_code LIKE '290%' OR
        icd_code LIKE '2941%' OR
        icd_code LIKE '2942%' OR
        icd_code = '2948' OR
        icd_code = '2949'
    ));

CREATE OR REPLACE TABLE psychiatric_icd_codes_suicide_self_harm AS
SELECT DISTINCT
    icd_version,
    icd_code,
    long_title
FROM psychiatric_icd_codes
WHERE
    lower(long_title) LIKE '%suicid%'
    OR lower(long_title) LIKE '%self-harm%'
    OR lower(long_title) LIKE '%self harm%'
    OR lower(long_title) LIKE '%intentional self%';

CREATE OR REPLACE TABLE psychiatric_icd_codes_other AS
SELECT DISTINCT
    icd_version,
    icd_code,
    long_title
FROM psychiatric_icd_codes
WHERE NOT EXISTS (
    SELECT 1
    FROM psychiatric_icd_codes_psychotic p
    WHERE p.icd_version = psychiatric_icd_codes.icd_version
      AND p.icd_code = psychiatric_icd_codes.icd_code
)
AND NOT EXISTS (
    SELECT 1
    FROM psychiatric_icd_codes_substance_related s
    WHERE s.icd_version = psychiatric_icd_codes.icd_version
      AND s.icd_code = psychiatric_icd_codes.icd_code
)
AND NOT EXISTS (
    SELECT 1
    FROM psychiatric_icd_codes_internalizing i
    WHERE i.icd_version = psychiatric_icd_codes.icd_version
      AND i.icd_code = psychiatric_icd_codes.icd_code
)
AND NOT EXISTS (
    SELECT 1
    FROM psychiatric_icd_codes_personality_behavioral pb
    WHERE pb.icd_version = psychiatric_icd_codes.icd_version
      AND pb.icd_code = psychiatric_icd_codes.icd_code
)
AND NOT EXISTS (
    SELECT 1
    FROM psychiatric_icd_codes_neurodevelopmental nd
    WHERE nd.icd_version = psychiatric_icd_codes.icd_version
      AND nd.icd_code = psychiatric_icd_codes.icd_code
)
AND NOT EXISTS (
    SELECT 1
    FROM psychiatric_icd_codes_neurocognitive nc
    WHERE nc.icd_version = psychiatric_icd_codes.icd_version
      AND nc.icd_code = psychiatric_icd_codes.icd_code
)
AND NOT EXISTS (
    SELECT 1
    FROM psychiatric_icd_codes_suicide_self_harm sh
    WHERE sh.icd_version = psychiatric_icd_codes.icd_version
      AND sh.icd_code = psychiatric_icd_codes.icd_code
);

SELECT 'psychotic' AS category, COUNT(*) AS n_codes FROM psychiatric_icd_codes_psychotic
UNION ALL
SELECT 'substance_related', COUNT(*) FROM psychiatric_icd_codes_substance_related
UNION ALL
SELECT 'internalizing', COUNT(*) FROM psychiatric_icd_codes_internalizing
UNION ALL
SELECT 'personality_behavioral', COUNT(*) FROM psychiatric_icd_codes_personality_behavioral
UNION ALL
SELECT 'neurodevelopmental', COUNT(*) FROM psychiatric_icd_codes_neurodevelopmental
UNION ALL
SELECT 'neurocognitive', COUNT(*) FROM psychiatric_icd_codes_neurocognitive
UNION ALL
SELECT 'self-harm', COUNT(*) FROM psychiatric_icd_codes_suicide_self_harm
UNION ALL
SELECT 'other', COUNT(*) FROM psychiatric_icd_codes_other
UNION ALL
SELECT 'all_psychiatric', COUNT(*) FROM psychiatric_icd_codes;