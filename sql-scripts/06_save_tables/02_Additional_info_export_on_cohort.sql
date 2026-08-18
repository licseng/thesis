-- 1. Demographic and admission descriptors
-- One row per matched admission
CREATE OR REPLACE TABLE export_matched_cohort_descriptors AS
WITH all_admissions_with_history AS (
    SELECT
        adm.*,

        ROW_NUMBER() OVER (
            PARTITION BY adm.subject_id
            ORDER BY adm.admittime, adm.hadm_id
        ) AS all_admission_order_for_subject,

        ROW_NUMBER() OVER (
            PARTITION BY adm.subject_id
            ORDER BY adm.admittime, adm.hadm_id
        ) - 1 AS n_prior_all_admissions_for_subject,

        SUM(
            CASE
                WHEN upper(adm.admission_type) LIKE '%EMER%'
                  OR upper(adm.admission_type) LIKE '%URGENT%'
                THEN 1 ELSE 0
            END
        ) OVER (
            PARTITION BY adm.subject_id
            ORDER BY adm.admittime, adm.hadm_id
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS n_prior_emergency_or_urgent_admissions_for_subject,

        LAG(adm.hadm_id) OVER (
            PARTITION BY adm.subject_id
            ORDER BY adm.admittime, adm.hadm_id
        ) AS previous_hadm_id_for_subject,

        LAG(adm.admittime) OVER (
            PARTITION BY adm.subject_id
            ORDER BY adm.admittime, adm.hadm_id
        ) AS previous_admittime_for_subject,

        LAG(adm.dischtime) OVER (
            PARTITION BY adm.subject_id
            ORDER BY adm.admittime, adm.hadm_id
        ) AS previous_dischtime_for_subject,

        FIRST_VALUE(adm.admittime) OVER (
            PARTITION BY adm.subject_id
            ORDER BY adm.admittime, adm.hadm_id
        ) AS first_observed_admittime_for_subject
    FROM admissions adm
),

all_admissions_with_prior_windows AS (
    SELECT
        curr.*,

        COALESCE(curr.n_prior_emergency_or_urgent_admissions_for_subject, 0)
            AS n_prior_emergency_or_urgent_admissions_for_subject_clean,

        DATE_DIFF('day', curr.previous_dischtime_for_subject, curr.admittime)
            AS days_since_previous_discharge_for_subject,

        DATE_DIFF('day', curr.first_observed_admittime_for_subject, curr.admittime)
            AS days_since_first_observed_admission_for_subject,

        COUNT(prev.hadm_id) FILTER (
            WHERE prev.admittime >= curr.admittime - INTERVAL '30 days'
              AND prev.admittime < curr.admittime
        ) AS n_prior_admissions_within_30d_for_subject,

        COUNT(prev.hadm_id) FILTER (
            WHERE prev.admittime >= curr.admittime - INTERVAL '90 days'
              AND prev.admittime < curr.admittime
        ) AS n_prior_admissions_within_90d_for_subject,

        COUNT(prev.hadm_id) FILTER (
            WHERE prev.admittime >= curr.admittime - INTERVAL '365 days'
              AND prev.admittime < curr.admittime
        ) AS n_prior_admissions_within_365d_for_subject

    FROM all_admissions_with_history curr

    LEFT JOIN admissions prev
        ON curr.subject_id = prev.subject_id
       AND prev.admittime < curr.admittime

    GROUP BY ALL
)

SELECT
    mc.*,

    pat.gender,
    pat.anchor_age,
    pat.anchor_year,
    pat.anchor_year_group,
    pat.dod,

    adm.admittime,
    adm.dischtime,
    adm.deathtime,
    adm.admission_type,
    adm.admission_location,
    adm.discharge_location,
    adm.insurance,
    adm.language,
    adm.race,
    adm.marital_status,
    adm.edregtime,
    adm.edouttime,
    adm.hospital_expire_flag,

    adm.all_admission_order_for_subject,
    adm.n_prior_all_admissions_for_subject,
    adm.n_prior_emergency_or_urgent_admissions_for_subject_clean
        AS n_prior_emergency_or_urgent_admissions_for_subject,
    adm.previous_hadm_id_for_subject,
    adm.previous_admittime_for_subject,
    adm.previous_dischtime_for_subject,
    adm.first_observed_admittime_for_subject,
    adm.days_since_previous_discharge_for_subject,
    adm.days_since_first_observed_admission_for_subject,
    adm.n_prior_admissions_within_30d_for_subject,
    adm.n_prior_admissions_within_90d_for_subject,
    adm.n_prior_admissions_within_365d_for_subject,
    CASE WHEN adm.n_prior_admissions_within_30d_for_subject > 0 THEN 1 ELSE 0 END
        AS has_prior_admission_within_30d_for_subject,
    CASE WHEN adm.n_prior_admissions_within_90d_for_subject > 0 THEN 1 ELSE 0 END
        AS has_prior_admission_within_90d_for_subject,
    CASE WHEN adm.n_prior_admissions_within_365d_for_subject > 0 THEN 1 ELSE 0 END
        AS has_prior_admission_within_365d_for_subject

FROM matched_cohort mc

LEFT JOIN patients pat
    ON mc.subject_id = pat.subject_id

LEFT JOIN all_admissions_with_prior_windows adm
    ON mc.subject_id = adm.subject_id
   AND mc.hadm_id = adm.hadm_id;


SELECT
    COUNT(*) AS n_rows,
    COUNT(DISTINCT hadm_id) AS n_admissions,
    COUNT(DISTINCT subject_id) AS n_subjects
FROM export_matched_cohort_descriptors;

-- 1b. Full admission history for all subjects in the matched cohort
-- One row per hospital admission for any subject appearing in matched_cohort.
-- This is not restricted to the matched admission itself; it is used to compute
-- true prior/readmission history in downstream Python analyses.
CREATE OR REPLACE TABLE export_matched_cohort_subject_admission_history AS
WITH matched_subjects AS (
    SELECT DISTINCT
        cohort,
        subject_id
    FROM matched_cohort
),

all_admissions_with_history AS (
    SELECT
        adm.*,

        ROW_NUMBER() OVER (
            PARTITION BY adm.subject_id
            ORDER BY adm.admittime, adm.hadm_id
        ) AS all_admission_order_for_subject,

        ROW_NUMBER() OVER (
            PARTITION BY adm.subject_id
            ORDER BY adm.admittime, adm.hadm_id
        ) - 1 AS n_prior_all_admissions_for_subject,

        SUM(
            CASE
                WHEN upper(adm.admission_type) LIKE '%EMER%'
                  OR upper(adm.admission_type) LIKE '%URGENT%'
                THEN 1 ELSE 0
            END
        ) OVER (
            PARTITION BY adm.subject_id
            ORDER BY adm.admittime, adm.hadm_id
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS n_prior_emergency_or_urgent_admissions_for_subject,

        LAG(adm.hadm_id) OVER (
            PARTITION BY adm.subject_id
            ORDER BY adm.admittime, adm.hadm_id
        ) AS previous_hadm_id_for_subject,

        LAG(adm.admittime) OVER (
            PARTITION BY adm.subject_id
            ORDER BY adm.admittime, adm.hadm_id
        ) AS previous_admittime_for_subject,

        LAG(adm.dischtime) OVER (
            PARTITION BY adm.subject_id
            ORDER BY adm.admittime, adm.hadm_id
        ) AS previous_dischtime_for_subject,

        FIRST_VALUE(adm.admittime) OVER (
            PARTITION BY adm.subject_id
            ORDER BY adm.admittime, adm.hadm_id
        ) AS first_observed_admittime_for_subject
    FROM admissions adm
),

all_admissions_with_prior_windows AS (
    SELECT
        curr.*,

        COALESCE(curr.n_prior_emergency_or_urgent_admissions_for_subject, 0)
            AS n_prior_emergency_or_urgent_admissions_for_subject_clean,

        DATE_DIFF('day', curr.previous_dischtime_for_subject, curr.admittime)
            AS days_since_previous_discharge_for_subject,

        DATE_DIFF('day', curr.first_observed_admittime_for_subject, curr.admittime)
            AS days_since_first_observed_admission_for_subject,

        COUNT(prev.hadm_id) FILTER (
            WHERE prev.admittime >= curr.admittime - INTERVAL '30 days'
              AND prev.admittime < curr.admittime
        ) AS n_prior_admissions_within_30d_for_subject,

        COUNT(prev.hadm_id) FILTER (
            WHERE prev.admittime >= curr.admittime - INTERVAL '90 days'
              AND prev.admittime < curr.admittime
        ) AS n_prior_admissions_within_90d_for_subject,

        COUNT(prev.hadm_id) FILTER (
            WHERE prev.admittime >= curr.admittime - INTERVAL '365 days'
              AND prev.admittime < curr.admittime
        ) AS n_prior_admissions_within_365d_for_subject

    FROM all_admissions_with_history curr

    LEFT JOIN admissions prev
        ON curr.subject_id = prev.subject_id
       AND prev.admittime < curr.admittime

    GROUP BY ALL
)

SELECT
    ms.cohort AS matched_cohort,
    ms.subject_id AS matched_subject_id,

    adm.subject_id,
    adm.hadm_id,
    adm.admittime,
    adm.dischtime,
    adm.deathtime,
    adm.admission_type,
    adm.admission_location,
    adm.discharge_location,
    adm.insurance,
    adm.language,
    adm.race,
    adm.marital_status,
    adm.edregtime,
    adm.edouttime,
    adm.hospital_expire_flag,

    pat.gender,
    pat.anchor_age,
    pat.anchor_year,
    pat.anchor_year_group,
    pat.dod,

    adm.all_admission_order_for_subject,
    adm.n_prior_all_admissions_for_subject,
    adm.n_prior_emergency_or_urgent_admissions_for_subject_clean
        AS n_prior_emergency_or_urgent_admissions_for_subject,
    adm.previous_hadm_id_for_subject,
    adm.previous_admittime_for_subject,
    adm.previous_dischtime_for_subject,
    adm.first_observed_admittime_for_subject,
    adm.days_since_previous_discharge_for_subject,
    adm.days_since_first_observed_admission_for_subject,
    adm.n_prior_admissions_within_30d_for_subject,
    adm.n_prior_admissions_within_90d_for_subject,
    adm.n_prior_admissions_within_365d_for_subject,
    CASE WHEN adm.n_prior_admissions_within_30d_for_subject > 0 THEN 1 ELSE 0 END
        AS has_prior_admission_within_30d_for_subject,
    CASE WHEN adm.n_prior_admissions_within_90d_for_subject > 0 THEN 1 ELSE 0 END
        AS has_prior_admission_within_90d_for_subject,
    CASE WHEN adm.n_prior_admissions_within_365d_for_subject > 0 THEN 1 ELSE 0 END
        AS has_prior_admission_within_365d_for_subject,

    CASE
        WHEN mc.hadm_id IS NOT NULL THEN 1 ELSE 0
    END AS is_matched_cohort_admission

FROM matched_subjects ms

JOIN all_admissions_with_prior_windows adm
    ON ms.subject_id = adm.subject_id

LEFT JOIN patients pat
    ON adm.subject_id = pat.subject_id

LEFT JOIN matched_cohort mc
    ON ms.cohort = mc.cohort
   AND adm.subject_id = mc.subject_id
   AND adm.hadm_id = mc.hadm_id;

SELECT
    COUNT(*) AS n_rows,
    COUNT(DISTINCT hadm_id) AS n_admissions,
    COUNT(DISTINCT subject_id) AS n_subjects,
    SUM(is_matched_cohort_admission) AS n_rows_that_are_matched_admissions
FROM export_matched_cohort_subject_admission_history;

-- 2. Laboratory events with readable lab names
-- All laboratory events belonging to matched admissions
CREATE OR REPLACE TABLE export_matched_cohort_labevents AS
SELECT
    mc.*,

    lab.labevent_id,
    lab.specimen_id,
    lab.itemid,
    lab.order_provider_id,
    lab.charttime,
    lab.storetime,
    lab.value,
    lab.valuenum,
    lab.valueuom,
    lab.ref_range_lower,
    lab.ref_range_upper,
    lab.flag,
    lab.priority,
    lab.comments,

    item.label AS lab_label,
    item.fluid AS lab_fluid,
    item.category AS lab_category

FROM matched_cohort mc

INNER JOIN labevents lab
    ON mc.subject_id = lab.subject_id
   AND mc.hadm_id = lab.hadm_id

LEFT JOIN d_labitems item
    ON lab.itemid = item.itemid;

--3. Microbiology events
-- All microbiology events belonging to matched admissions
CREATE OR REPLACE TABLE export_matched_cohort_microbiologyevents AS
SELECT
    mc.*,
    micro.* EXCLUDE (subject_id, hadm_id)

FROM matched_cohort mc

INNER JOIN microbiologyevents micro
    ON mc.subject_id = micro.subject_id
   AND mc.hadm_id = micro.hadm_id;

--4. Provider orders
-- Provider orders belonging to matched admissions
CREATE OR REPLACE TABLE export_matched_cohort_poe AS
SELECT
    mc.*,
    p.* EXCLUDE (subject_id, hadm_id)

FROM matched_cohort mc

INNER JOIN poe p
    ON mc.subject_id = p.subject_id
   AND mc.hadm_id = p.hadm_id;

--5. Provider order details 
-- Additional details for orders belonging to matched admissions
CREATE OR REPLACE TABLE export_matched_cohort_poe_detail AS
SELECT
    mc.*,

    p.poe_id,
    p.poe_seq,
    p.ordertime,
    p.order_type,
    p.order_subtype,
    p.transaction_type,
    p.discontinue_of_poe_id,
    p.discontinued_by_poe_id,
    p.order_provider_id,
    p.order_status,

    pd.field_name,
    pd.field_value

FROM matched_cohort mc

INNER JOIN poe p
    ON mc.subject_id = p.subject_id
   AND mc.hadm_id = p.hadm_id

INNER JOIN poe_detail pd
    ON p.poe_id = pd.poe_id;

--6. icd codes for the cohort
CREATE OR REPLACE TABLE export_matched_cohort_diagnoses AS
SELECT
    mc.cohort,
    mc.matched_role,
    mc.pair_id,
    mc.subject_id,
    mc.hadm_id,

    d.seq_num,
    d.icd_version,
    d.icd_code,
    dd.long_title,

    CASE
        WHEN p.icd_code IS NOT NULL THEN 1 ELSE 0
    END AS is_psychiatric_icd,

    CASE
        WHEN g.icd_code IS NOT NULL THEN 1 ELSE 0
    END AS is_grey_zone_physical_icd,

    CASE
        WHEN d.seq_num = 1 THEN 1 ELSE 0
    END AS is_primary_diagnosis

FROM matched_cohort mc

JOIN diagnoses_icd d
    ON mc.subject_id = d.subject_id
   AND mc.hadm_id = d.hadm_id

LEFT JOIN d_icd_diagnoses dd
    ON d.icd_version = dd.icd_version
   AND d.icd_code = dd.icd_code

LEFT JOIN psychiatric_icd_codes p
    ON d.icd_version = p.icd_version
   AND d.icd_code = p.icd_code

LEFT JOIN grey_zone_physical_icd_codes g
    ON d.icd_version = g.icd_version
   AND d.icd_code = g.icd_code;
