--Same-admission ONLY MHC1!!! MHC1-sa
CREATE OR REPLACE TABLE possible_overshadowing_admissions_psychiatric_current_only AS
SELECT *
FROM possible_overshadowing_admissions_psychiatric
WHERE has_secondary_psychiatric_same_admission = 1;


CREATE OR REPLACE TABLE possible_overshadowing_admissions_psychiatric_focus4 AS
SELECT DISTINCT
    p.*
FROM possible_overshadowing_admissions_psychiatric_current_only p
JOIN diagnoses_icd d
    ON p.subject_id = d.subject_id
   AND p.hadm_id = d.hadm_id
WHERE d.seq_num > 1
  AND (
      EXISTS (
          SELECT 1
          FROM psychiatric_icd_codes_internalizing c
          WHERE d.icd_version = c.icd_version
            AND d.icd_code = c.icd_code
      )
      OR EXISTS (
          SELECT 1
          FROM psychiatric_icd_codes_substance_related c
          WHERE d.icd_version = c.icd_version
            AND d.icd_code = c.icd_code
      )
      OR EXISTS (
          SELECT 1
          FROM psychiatric_icd_codes_psychotic c
          WHERE d.icd_version = c.icd_version
            AND d.icd_code = c.icd_code
      )
      OR EXISTS (
          SELECT 1
          FROM psychiatric_icd_codes_personality_behavioral c
          WHERE d.icd_version = c.icd_version
            AND d.icd_code = c.icd_code
      )
  );