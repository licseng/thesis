--All the admissions where there were psychiatric related icd codes (with the 4 focus group) in a previous admission
CREATE OR REPLACE TABLE possible_overshadowing_admissions_psychiatric_MHH_focus4 AS
SELECT DISTINCT
    p.*
FROM possible_overshadowing_admissions_psychiatric p
WHERE p.has_prior_psychiatric_history = 1
  AND (
      lower(p.prior_psych_categories) LIKE '%internalizing%'
      OR lower(p.prior_psych_categories) LIKE '%substance_related%'
      OR lower(p.prior_psych_categories) LIKE '%psychotic%'
      OR lower(p.prior_psych_categories) LIKE '%personality_behavioral%'
  );