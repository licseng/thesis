-- 1. Demographic and admission descriptors
-- One row per matched admission
CREATE OR REPLACE TABLE export_matched_cohort_descriptors AS
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
    adm.hospital_expire_flag

FROM matched_cohort mc

LEFT JOIN patients pat
    ON mc.subject_id = pat.subject_id

LEFT JOIN admissions adm
    ON mc.subject_id = adm.subject_id
   AND mc.hadm_id = adm.hadm_id;


SELECT
    COUNT(*) AS n_rows,
    COUNT(DISTINCT hadm_id) AS n_admissions,
    COUNT(DISTINCT subject_id) AS n_subjects
FROM export_matched_cohort_descriptors;

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