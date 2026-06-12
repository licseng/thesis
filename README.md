# Thesis Repository

## SQL scripts

This repository contains the SQL workflow used to construct the thesis cohorts from MIMIC-IV.

The SQL scripts:

- define psychiatric ICD code groups

- build a clean non-psychiatric main admission base dataset with discharge notes and icd labels, excluding some ambiguous admissions

- define admission-level mental health context (**MHC**)

- separate subjects into:

  - **MHC0**: pure control group

  - **MHC1**: pure case group with same-admission and/or history-only psychiatric context

  - **MHC_0_to_1**: mixed transition group for trajectory analysis
 
- further clean **MHC1** to **MHC1-sa (same-admission)** 

- create export tables for downstream analysis

The main case-control comparison uses:

- **Control:** `MHC0`

- **Case:** `MHC1-sa`
