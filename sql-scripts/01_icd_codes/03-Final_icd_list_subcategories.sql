-- Assign every psychiatric ICD code to exactly one diagnostic subgroup.
--
-- Precedence matters. The revised psychotic-illness definition is evaluated
-- first so that mood disorders with psychosis and postpartum psychosis are not
-- also assigned to internalizing or personality/behavioral. This prevents one
-- ICD code from creating an artificial comorbidity. An admission or subject
-- can still have several subgroups when genuinely different diagnosis codes
-- are present.

CREATE OR REPLACE TABLE psychiatric_icd_code_subcategories AS
SELECT DISTINCT
    p.icd_version,
    p.icd_code,
    p.long_title,
    CASE
        WHEN EXISTS (
            SELECT 1
            FROM psychosis_icd_codes_extended psy
            WHERE psy.icd_version = p.icd_version
              AND psy.icd_code = p.icd_code
        ) THEN 'psychotic'

        WHEN
            (p.icd_version = 10 AND (
                p.icd_code LIKE 'F10%' OR
                p.icd_code LIKE 'F11%' OR
                p.icd_code LIKE 'F12%' OR
                p.icd_code LIKE 'F13%' OR
                p.icd_code LIKE 'F14%' OR
                p.icd_code LIKE 'F15%' OR
                p.icd_code LIKE 'F16%' OR
                p.icd_code LIKE 'F17%' OR
                p.icd_code LIKE 'F18%' OR
                p.icd_code LIKE 'F19%'
            ))
            OR
            (p.icd_version = 9 AND (
                p.icd_code LIKE '291%' OR
                p.icd_code LIKE '292%' OR
                p.icd_code LIKE '303%' OR
                p.icd_code LIKE '304%' OR
                p.icd_code LIKE '305%' OR
                p.icd_code = 'V113'
            ))
            THEN 'substance_related'

        WHEN
            (p.icd_version = 10 AND (
                p.icd_code LIKE 'F30%' OR
                p.icd_code LIKE 'F31%' OR
                p.icd_code LIKE 'F32%' OR
                p.icd_code LIKE 'F33%' OR
                p.icd_code LIKE 'F34%' OR
                p.icd_code LIKE 'F38%' OR
                p.icd_code LIKE 'F39%' OR
                p.icd_code LIKE 'F40%' OR
                p.icd_code LIKE 'F41%' OR
                p.icd_code LIKE 'F42%' OR
                p.icd_code LIKE 'F43%' OR
                p.icd_code LIKE 'F44%' OR
                p.icd_code LIKE 'F45%' OR
                p.icd_code LIKE 'F48%' OR
                p.icd_code LIKE 'F50%'
            ))
            OR
            (p.icd_version = 9 AND (
                p.icd_code LIKE '296%' OR
                p.icd_code LIKE '300%' OR
                p.icd_code LIKE '306%' OR
                p.icd_code LIKE '308%' OR
                p.icd_code LIKE '309%' OR
                p.icd_code = '311' OR
                p.icd_code LIKE '3071%' OR
                p.icd_code = '3130' OR
                p.icd_code = '3131' OR
                p.icd_code LIKE '3132%' OR
                p.icd_code = 'V111' OR
                p.icd_code = 'V112' OR
                p.icd_code = 'V114'
            ))
            THEN 'internalizing'

        WHEN
            (p.icd_version = 10 AND (
                p.icd_code LIKE 'F60%' OR
                p.icd_code LIKE 'F61%' OR
                p.icd_code LIKE 'F62%' OR
                p.icd_code LIKE 'F63%' OR
                p.icd_code LIKE 'F64%' OR
                p.icd_code LIKE 'F65%' OR
                p.icd_code LIKE 'F66%' OR
                p.icd_code LIKE 'F68%' OR
                p.icd_code LIKE 'F69%' OR
                p.icd_code LIKE 'F51%' OR
                p.icd_code LIKE 'F52%' OR
                p.icd_code LIKE 'F53%' OR
                p.icd_code LIKE 'F54%' OR
                p.icd_code LIKE 'F55%' OR
                p.icd_code LIKE 'F59%'
            ))
            OR
            (p.icd_version = 9 AND (
                p.icd_code LIKE '301%' OR
                p.icd_code LIKE '302%' OR
                p.icd_code LIKE '3070%' OR
                p.icd_code LIKE '3073%' OR
                p.icd_code LIKE '3074%' OR
                p.icd_code LIKE '3075%' OR
                p.icd_code LIKE '3076%' OR
                p.icd_code LIKE '3077%' OR
                p.icd_code LIKE '3078%' OR
                p.icd_code LIKE '3079%' OR
                p.icd_code LIKE '312%' OR
                p.icd_code LIKE '3133%' OR
                p.icd_code LIKE '3138%' OR
                p.icd_code = '3139'
            ))
            THEN 'personality_behavioral'

        WHEN
            (p.icd_version = 10 AND (
                p.icd_code LIKE 'F70%' OR
                p.icd_code LIKE 'F71%' OR
                p.icd_code LIKE 'F72%' OR
                p.icd_code LIKE 'F73%' OR
                p.icd_code LIKE 'F78%' OR
                p.icd_code LIKE 'F79%' OR
                p.icd_code LIKE 'F80%' OR
                p.icd_code LIKE 'F81%' OR
                p.icd_code LIKE 'F82%' OR
                p.icd_code LIKE 'F83%' OR
                p.icd_code LIKE 'F84%' OR
                p.icd_code LIKE 'F88%' OR
                p.icd_code LIKE 'F89%' OR
                p.icd_code LIKE 'F90%' OR
                p.icd_code LIKE 'F91%' OR
                p.icd_code LIKE 'F92%' OR
                p.icd_code LIKE 'F93%' OR
                p.icd_code LIKE 'F94%' OR
                p.icd_code LIKE 'F95%' OR
                p.icd_code LIKE 'F98%'
            ))
            OR
            (p.icd_version = 9 AND (
                p.icd_code LIKE '299%' OR
                p.icd_code LIKE '3072%' OR
                p.icd_code LIKE '314%' OR
                p.icd_code LIKE '315%' OR
                p.icd_code LIKE '317%' OR
                p.icd_code LIKE '318%' OR
                p.icd_code LIKE '319%'
            ))
            THEN 'neurodevelopmental'

        WHEN
            (p.icd_version = 10 AND (
                p.icd_code LIKE 'F01%' OR
                p.icd_code LIKE 'F02%' OR
                p.icd_code LIKE 'F03%'
            ))
            OR
            (p.icd_version = 9 AND (
                p.icd_code LIKE '290%' OR
                p.icd_code LIKE '2941%' OR
                p.icd_code LIKE '2942%' OR
                p.icd_code = '2948' OR
                p.icd_code = '2949'
            ))
            THEN 'neurocognitive'

        WHEN lower(p.long_title) LIKE '%suicid%'
          OR lower(p.long_title) LIKE '%self-harm%'
          OR lower(p.long_title) LIKE '%self harm%'
          OR lower(p.long_title) LIKE '%intentional self%'
            THEN 'suicide_self_harm'

        ELSE 'other'
    END AS psych_category
FROM psychiatric_icd_codes p;


-- Compatibility tables retained because downstream cohort/statistics scripts
-- already use these names. All are derived from the exclusive mapping above.
CREATE OR REPLACE TABLE psychiatric_icd_codes_psychotic AS
SELECT icd_version, icd_code, long_title
FROM psychiatric_icd_code_subcategories
WHERE psych_category = 'psychotic';

CREATE OR REPLACE TABLE psychiatric_icd_codes_substance_related AS
SELECT icd_version, icd_code, long_title
FROM psychiatric_icd_code_subcategories
WHERE psych_category = 'substance_related';

CREATE OR REPLACE TABLE psychiatric_icd_codes_internalizing AS
SELECT icd_version, icd_code, long_title
FROM psychiatric_icd_code_subcategories
WHERE psych_category = 'internalizing';

CREATE OR REPLACE TABLE psychiatric_icd_codes_personality_behavioral AS
SELECT icd_version, icd_code, long_title
FROM psychiatric_icd_code_subcategories
WHERE psych_category = 'personality_behavioral';

CREATE OR REPLACE TABLE psychiatric_icd_codes_neurodevelopmental AS
SELECT icd_version, icd_code, long_title
FROM psychiatric_icd_code_subcategories
WHERE psych_category = 'neurodevelopmental';

CREATE OR REPLACE TABLE psychiatric_icd_codes_neurocognitive AS
SELECT icd_version, icd_code, long_title
FROM psychiatric_icd_code_subcategories
WHERE psych_category = 'neurocognitive';

CREATE OR REPLACE TABLE psychiatric_icd_codes_suicide_self_harm AS
SELECT icd_version, icd_code, long_title
FROM psychiatric_icd_code_subcategories
WHERE psych_category = 'suicide_self_harm';

CREATE OR REPLACE TABLE psychiatric_icd_codes_other AS
SELECT icd_version, icd_code, long_title
FROM psychiatric_icd_code_subcategories
WHERE psych_category = 'other';


-- Overall code counts under the revised mutually exclusive definition.
SELECT
    psych_category AS category,
    COUNT(*) AS n_codes
FROM psychiatric_icd_code_subcategories
GROUP BY psych_category

UNION ALL

SELECT
    'all_psychiatric' AS category,
    COUNT(*) AS n_codes
FROM psychiatric_icd_code_subcategories

ORDER BY category;


-- Integrity check 1: this should return one row with identical totals.
SELECT
    (SELECT COUNT(*) FROM psychiatric_icd_codes) AS n_source_codes,
    (SELECT COUNT(*) FROM psychiatric_icd_code_subcategories) AS n_assigned_codes,
    (SELECT COUNT(DISTINCT concat(icd_version, ':', icd_code))
     FROM psychiatric_icd_code_subcategories) AS n_distinct_assigned_codes;


-- Integrity check 2: this should return zero rows. It independently verifies
-- that no code occurs in more than one downstream compatibility table.
WITH downstream_assignments AS (
    SELECT icd_version, icd_code, 'psychotic' AS category
    FROM psychiatric_icd_codes_psychotic
    UNION ALL
    SELECT icd_version, icd_code, 'substance_related'
    FROM psychiatric_icd_codes_substance_related
    UNION ALL
    SELECT icd_version, icd_code, 'internalizing'
    FROM psychiatric_icd_codes_internalizing
    UNION ALL
    SELECT icd_version, icd_code, 'personality_behavioral'
    FROM psychiatric_icd_codes_personality_behavioral
    UNION ALL
    SELECT icd_version, icd_code, 'neurodevelopmental'
    FROM psychiatric_icd_codes_neurodevelopmental
    UNION ALL
    SELECT icd_version, icd_code, 'neurocognitive'
    FROM psychiatric_icd_codes_neurocognitive
    UNION ALL
    SELECT icd_version, icd_code, 'suicide_self_harm'
    FROM psychiatric_icd_codes_suicide_self_harm
    UNION ALL
    SELECT icd_version, icd_code, 'other'
    FROM psychiatric_icd_codes_other
)
SELECT
    icd_version,
    icd_code,
    COUNT(*) AS n_categories,
    string_agg(category, ' | ' ORDER BY category) AS categories
FROM downstream_assignments
GROUP BY icd_version, icd_code
HAVING COUNT(*) > 1
ORDER BY icd_version, icd_code;
