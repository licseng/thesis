CREATE OR REPLACE TABLE psychosis_icd_codes_extended AS
SELECT
    icd_version,
    icd_code,
    long_title
FROM d_icd_diagnoses d
WHERE
(
    d.icd_version = 10 AND (
        d.icd_code LIKE 'F20%' OR
        d.icd_code LIKE 'F22%' OR
        d.icd_code LIKE 'F23%' OR
        d.icd_code LIKE 'F24%' OR
        d.icd_code LIKE 'F25%' OR
        d.icd_code LIKE 'F28%' OR
        d.icd_code LIKE 'F29%' OR
        d.icd_code LIKE 'F060%' OR
        d.icd_code LIKE 'F061%' OR
        d.icd_code LIKE 'F062%' OR
        d.icd_code LIKE 'F531%' OR
        d.icd_code LIKE 'F10%5%' OR
        d.icd_code LIKE 'F11%5%' OR
        d.icd_code LIKE 'F12%5%' OR
        d.icd_code LIKE 'F13%5%' OR
        d.icd_code LIKE 'F14%5%' OR
        d.icd_code LIKE 'F15%5%' OR
        d.icd_code LIKE 'F16%5%' OR
        d.icd_code LIKE 'F17%5%' OR
        d.icd_code LIKE 'F18%5%' OR
        d.icd_code LIKE 'F19%5%' OR
        d.icd_code = 'F0152' OR
        d.icd_code = 'F01A2' OR
        d.icd_code = 'F01B2' OR
        d.icd_code = 'F01C2' OR
        d.icd_code = 'F0282' OR
        d.icd_code = 'F02A2' OR
        d.icd_code = 'F02B2' OR
        d.icd_code = 'F02C2' OR
        d.icd_code = 'F0392' OR
        d.icd_code = 'F03A2' OR
        d.icd_code = 'F03B2' OR
        d.icd_code = 'F03C2' OR
        d.icd_code = 'F302' OR
        d.icd_code = 'F312' OR
        d.icd_code = 'F315' OR
        d.icd_code = 'F3164' OR
        d.icd_code = 'F323' OR
        d.icd_code = 'F333'
    )
)
OR
(
    d.icd_version = 9 AND (
        d.icd_code LIKE '295%' OR
        d.icd_code = '2971' OR
        d.icd_code = '2973' OR
        d.icd_code = '2980' OR
        d.icd_code = '2981' OR
        d.icd_code = '2984' OR
        d.icd_code = '2988' OR
        d.icd_code = '2989' OR
        d.icd_code = '29012' OR
        d.icd_code = '29020' OR
        d.icd_code = '29042' OR
        d.icd_code = '29381' OR
        d.icd_code = '29382' OR
        d.icd_code = '29211' OR
        d.icd_code = '29212' OR
        d.icd_code = '2913' OR
        d.icd_code = '2915' OR
        d.icd_code = '29604' OR
        d.icd_code = '29614' OR
        d.icd_code = '29624' OR
        d.icd_code = '29634' OR
        d.icd_code = '29644' OR
        d.icd_code = '29654' OR
        d.icd_code = '29664' OR
        d.icd_code = 'V110' OR
        d.icd_code = '2908' OR
        d.icd_code = '2909'
    )
)
ORDER BY icd_version, icd_code;