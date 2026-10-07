# Pipeline audit — 2026-10-07

Scope: finalized SQL cohort construction and Python stages 01–04, followed by
inspection of the existing annotation/classification/sentiment entry points.
This was a code and metadata/aggregate audit, not a rerun of the pipeline or an
inspection of clinical free text, laboratory values or classifier evidence.
No SQL tables or analytical results were changed.

## Implemented workflow

1. Finalized SQL 01 constructs the physical-admission base: discharge note and
   primary ICD required; psychiatric and grey-zone primary diagnoses excluded.
2. SQL 02 defines trajectory-wide psychiatric-code-negative MHC0 and MHC1 with
   prior psychiatric coding and/or current secondary psychiatric coding.
3. SQL 03 applies the extended psychosis list and symmetrically excludes patients
   with more than 50 total MIMIC hospital admissions. SQL 04 exports matching
   inputs. SQL 05 describes the eligible pool, not the matched analytic cohort.
4. Python 01 extracts, cleans and filters chief complaints, generates QuickUMLS
   concepts, and embeds retained complaints. Final filtering also excludes
   negated psychiatric/substance/self-harm TargetMatcher hits; this is a
   conservative eligibility rule, not proof of an active psychiatric complaint.
5. Python 02 computes Elixhauser scores, matches admissions, assesses balance and
   matching selection, exports IDs for SQL 06, and characterizes matched cohorts.
6. Matched notes are section-parsed; Python 03 computes complexity and keyword
   analyses, including language/race and neurocognitive sensitivity outputs.
7. Python 04 retains complete pairs in the five selected pure CC groups and
   analyzes laboratory measurements/specimens, microbiology and POE transactions,
   hospital/ED duration, in-hospital mortality and 365-day post-discharge mortality.

## Checks performed on the current saved artifacts

| Check | Result |
|---|---|
| Final usable chief complaints | MHC1-psychosis 7,150; MHC0 114,498 |
| Embeddings | 768 dimensions; correct sorted ID mapping, finite, unit-normalized |
| Matching-variable embedding indices | Correct admission-to-matrix-row mapping in both cohorts |
| Matching | 6,432 pairs; no duplicate admissions or pair IDs; no patient overlap |
| Matching restrictions | Sex/insurance-group agreement; age-band distance ≤1; cosine >0.90; Elixhauser difference ≤5 |
| SQL matched IDs and descriptors | Same 12,864 admissions as current Python matching; no duplicates |
| SQL psychosis-context export | Same 6,432 case admissions |
| Parsed notes | 6,432 per cohort, exactly matching the corresponding admissions |
| Language-model covariates | Same 12,864 matched admissions; no duplicate admissions |
| Five-CC clinical selection | 1,993 complete pairs, all from the current matched cohort |
| Laboratory recovery | Corrected totals incorporated: 8,595 rows across 557 admissions |
| Primary clinical saved-input checksums | Match saved inputs for full-stay event, stay/mortality and post-discharge models |
| Primary hospital-outcome models | 40 fits, all numerically accepted; six sparse-mortality cautions remain |
| Python syntax | No syntax errors detected |
| Rename | Folder and references updated to `04_clinical_route_analyses`; generated outputs remain ignored |

These checks do not prove statistical assumptions, parser accuracy or clinical
classification validity. They establish specific consistency properties of the
current saved artifacts; no end-to-end rebuild was performed.

## Before a full LLM classification run

1. **Implement the agreed one-stage diagnostic-overshadowing route.**
   `06_classification/02_classification_diag_overshadowing/01_run_diagnostic_overshadowing_section_classifier.py`
   still requires first-stage psychiatric-classifier results and sends only
   positive sections. It therefore does not implement the decision to drop
   psychiatric integration as a separate gate.
2. **Fix the input scope and denominator before interpreting classifications.**
   The existing psychiatric prefilter includes only MHC1 notes with psychiatric
   keywords and excludes several background sections by default. This can be
   appropriate for a case-only candidate analysis, but cannot by itself support
   an all-note MHC0/MHC1 comparison. Unscreened or excluded sections are not
   automatically classifier-negative. Choose the text context appropriate to
   each task before preparing the new inputs.
3. **Finalize prompts and validate a pilot before treating labels as outcomes.**
   No completed annotation exports or annotation guidelines were found in the
   active folder 05. No dedicated contextual SL classifier was found; the
   existing SL outputs remain keyword candidates. Include appropriate negative
   examples in validation, not only positive candidates. Sentiment is a separate
   outcome and does not establish stigma. Diagnostic-overshadowing labels should
   concern documented evidence of possible overshadowing, not confirmed missed
   care inferred from a discharge note alone.
4. **Keep new runs separate and make resumption configuration-safe.**
   The diagnostic classifier reassigns sequential `classifier_row_id` values
   after selection and resumes by those IDs, without enforcing matching input,
   prompt and model hashes. Reusing an output directory after changing selection
   or prompt could silently reuse wrong results. Use a fresh run directory and
   add a run-signature check before relying on automatic resume. Default pilot
   limits are ten notes. The diagnostic script supports the institute's
   OpenAI-compatible interface; the sentiment script currently supports only
   Ollama/Transformers, so its backend also needs aligning if that API is intended.

## Existing analysis qualifications — not reasons to restart the pipeline

- Seven of 152 M0–M3 language-complexity fits are singular, all for words per
  sentence in brief hospital course/present illness. Inspect the near-zero
  variance components and report the qualification. The current R code records
  variance/convergence information but does not implement residual/influence
  diagnostics; targeted checks remain useful before final thesis reporting.
- Numerical acceptance is not a guarantee of reliable mortality inference.
  Sparse-death and influence cautions remain relevant. Mortality models are
  admission-based; clustering does not turn their denominator into patients.
  Unique-patient death counts, if wanted, are a separate descriptive summary.
  A single recorded death may label several overlapping post-discharge windows.
- Clinical late-stay comparisons concern patients still hospitalized at those
  times. Event counts are documentation/activity proxies, not direct measures of
  care quality or diagnostic overshadowing. POE transactions are not necessarily
  unique tests or unique clinical decisions; `poe_detail` is not a primary outcome.
- “History” is operationalized by an earlier admission time with a coded
  diagnosis. This does not prove that the code was available to clinicians when
  a particular note was written. MHC0 is code-negative in the observed MIMIC
  trajectory, not necessarily psychiatrically healthy throughout life.
- Preserve the existing FDR families when presenting reduced tables. Omitting
  metrics from a thesis table is not a reason to recalculate FDR over only those
  displayed metrics.
- Several older scripts/results remain intentionally available: folder-04
  script 04 uses the earlier nausea-including subgroup design; script 07 reads
  the legacy folder-09 mortality dataset. Do not mistake these for required next
  steps or current primary results.

Conclusion: the checked stages 01–04 do not reveal a missing major processing
stage requiring a restart. The next step is to finalize LLM inputs, prompts and
run safeguards and validate a pilot—not to launch the unchanged legacy
classification chain over the full cohort.
