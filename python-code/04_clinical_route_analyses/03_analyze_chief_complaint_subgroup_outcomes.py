"""Describe work-up and stay duration within five complete pure-CC pairs.

Run 01_describe_chief_complaint_subgroups.py first. The primary summaries retain
both members of an original matched pair with valid values for each outcome.
Missing/negative stay durations are excluded per outcome, without converting
missing event counts to zero. Event counts represent recorded database rows.
Optional negative-binomial inference uses crossed patient/pair random intercepts.
The --fit-whole-cohort-stay-mortality mode uses all matched admissions for
LOS/ED stay and both mortality endpoints, without pooling clinical work-up counts.
No clinical free-text export is performed.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import subprocess
import sys

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
COHORT_MATCHING_DIR = PROJECT_DIR / "02_cohort_matching"
sys.path.insert(0, str(COHORT_MATCHING_DIR))

import _matched_cohort_characterization_common as common  # noqa: E402


ASSIGNMENT_PATH = (
    SCRIPT_DIR
    / "analysis_output_chief_complaint_subgroup_balance_check"
    / "chief_complaint_subgroup_admission_assignments.csv"
)
UTILIZATION_COUNTS_PATH = (
    COHORT_MATCHING_DIR
    / "analysis_output_matched_cohort_characterization_admission_level"
    / "matched_cohort_utilization_counts_by_admission.csv"
)
OUTPUT_DIR = SCRIPT_DIR / "analysis_output_chief_complaint_subgroup_outcomes"

ID_COLUMNS = ["cohort", "subject_id", "hadm_id"]
GROUP_COLUMN = "exclusive_combined_group"
GROUP_LABELS = {
    "abdominal_pain_nausea_vomiting": "Abdominal pain / nausea / vomiting",
    "chest_pain_shortness_of_breath": "Chest pain / shortness of breath",
}
CATEGORICAL_COLUMNS = [
    "admission_type",
    "admission_location",
    "discharge_location",
    "insurance",
    "race_group",
    "ethnicity_from_race",
    "language",
    "marital_status",
]
NUMERIC_SUMMARY_COLUMNS = [
    "age_at_admission",
    "elixhauser_score",
    "hospital_los_days",
    "ed_los_hours",
    "n_labevents_rows",
    "n_microbiologyevents_rows",
    "n_poe_rows",
    "n_poe_detail_rows",
]


def load_subgroup_assignments() -> pd.DataFrame:
    """Load admission-level subgroup flags and keep exclusive subgroup rows."""
    if not ASSIGNMENT_PATH.exists():
        raise FileNotFoundError(
            "Missing subgroup assignments. Run "
            "04_clinical_route_analyses/01_describe_chief_complaint_subgroups.py first: "
            f"{ASSIGNMENT_PATH}"
        )

    assignments = pd.read_csv(ASSIGNMENT_PATH)
    missing = sorted(set(ID_COLUMNS + ["pair_id", GROUP_COLUMN]) - set(assignments.columns))
    if missing:
        raise ValueError(f"Subgroup assignment file is missing columns: {missing}")

    assignments = assignments.copy()
    assignments["cohort"] = assignments["cohort"].astype("string").str.strip()
    assignments["subject_id"] = pd.to_numeric(
        assignments["subject_id"], errors="raise"
    ).astype(int)
    assignments["hadm_id"] = pd.to_numeric(assignments["hadm_id"], errors="raise").astype(int)
    assignments[GROUP_COLUMN] = assignments[GROUP_COLUMN].astype("string").str.strip()
    assignments = assignments.loc[
        assignments[GROUP_COLUMN].isin(GROUP_LABELS)
    ].copy()
    assignments["exclusive_combined_group_label"] = assignments[GROUP_COLUMN].map(
        GROUP_LABELS
    )
    return assignments


def load_descriptors() -> pd.DataFrame:
    """Load descriptor rows and derive LOS/death fields used in summaries."""
    descriptors = common.add_derived_descriptor_columns(
        common.validate_id_columns(
            common.load_required_table("descriptors"),
            "descriptors",
        )
    )
    if descriptors.duplicated(ID_COLUMNS).any():
        duplicated = int(descriptors.duplicated(ID_COLUMNS, keep=False).sum())
        raise ValueError(
            f"Descriptor table has {duplicated} duplicated admission ID rows."
        )

    descriptors = descriptors.copy()
    if "admittime" in descriptors.columns and "dischtime" in descriptors.columns:
        descriptors["admittime"] = pd.to_datetime(descriptors["admittime"], errors="coerce")
        descriptors["dischtime"] = pd.to_datetime(descriptors["dischtime"], errors="coerce")
        descriptors["hospital_los_days"] = (
            descriptors["dischtime"] - descriptors["admittime"]
        ).dt.total_seconds() / 86400

    if "edregtime" in descriptors.columns and "edouttime" in descriptors.columns:
        descriptors["edregtime"] = pd.to_datetime(descriptors["edregtime"], errors="coerce")
        descriptors["edouttime"] = pd.to_datetime(descriptors["edouttime"], errors="coerce")
        descriptors["ed_los_hours"] = (
            descriptors["edouttime"] - descriptors["edregtime"]
        ).dt.total_seconds() / 3600

    if "deathtime" in descriptors.columns:
        descriptors["has_deathtime"] = pd.to_datetime(
            descriptors["deathtime"], errors="coerce"
        ).notna()
    if "dod" in descriptors.columns:
        descriptors["has_dod"] = pd.to_datetime(descriptors["dod"], errors="coerce").notna()
    if "hospital_expire_flag" in descriptors.columns:
        descriptors["hospital_expire_flag"] = pd.to_numeric(
            descriptors["hospital_expire_flag"], errors="coerce"
        )
        descriptors["died_in_hospital"] = descriptors["hospital_expire_flag"].eq(1)

    return descriptors


def load_or_build_utilization_counts() -> pd.DataFrame:
    """Load cached admission-level utilization counts, or rebuild them."""
    if UTILIZATION_COUNTS_PATH.exists():
        counts = pd.read_csv(UTILIZATION_COUNTS_PATH)
        return common.validate_id_columns(counts, "utilization_counts")

    matched_ids = common.load_expected_matched_ids()
    event_tables = {
        "labevents": common.load_optional_table("labevents"),
        "microbiologyevents": common.load_optional_table("microbiologyevents"),
        "poe": common.load_optional_table("poe"),
        "poe_detail": common.load_optional_table("poe_detail"),
    }
    return common.build_event_counts_by_admission(matched_ids, event_tables)


def build_analysis_dataset(
    assignments: pd.DataFrame,
    descriptors: pd.DataFrame,
    utilization_counts: pd.DataFrame,
) -> pd.DataFrame:
    """Join subgroup assignments to descriptors and utilization counts."""
    descriptor_columns = [
        column
        for column in [
            *ID_COLUMNS,
            "pair_id",
            "matched_role",
            "gender",
            "admittime",
            "dischtime",
            "deathtime",
            "dod",
            "admission_type",
            "admission_location",
            "discharge_location",
            "insurance",
            "language",
            "race",
            "race_group",
            "ethnicity_from_race",
            "marital_status",
            "hospital_expire_flag",
            "died_in_hospital",
            "has_deathtime",
            "has_dod",
            "hospital_los_days",
            "ed_los_hours",
        ]
        if column in descriptors.columns
    ]
    analysis = assignments.merge(
        descriptors.loc[:, descriptor_columns],
        on=ID_COLUMNS,
        how="left",
        validate="one_to_one",
        suffixes=("", "_descriptor"),
    )

    count_columns = [
        column
        for column in utilization_counts.columns
        if column in ID_COLUMNS or column.startswith("n_")
    ]
    analysis = analysis.merge(
        utilization_counts.loc[:, count_columns],
        on=ID_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    for column in [column for column in analysis.columns if column.startswith("n_")]:
        analysis[column] = analysis[column].fillna(0).astype(int)

    if "age_at_admission" not in analysis.columns and "anchor_age" in descriptors.columns:
        age = descriptors.loc[:, ID_COLUMNS + ["anchor_age"]].rename(
            columns={"anchor_age": "age_at_admission"}
        )
        analysis = analysis.merge(age, on=ID_COLUMNS, how="left", validate="one_to_one")

    keep_columns = [
        column
        for column in [
            "pair_id",
            *ID_COLUMNS,
            GROUP_COLUMN,
            "exclusive_combined_group_label",
            "sex",
            "gender",
            "insurance_group",
            "age_at_admission",
            "elixhauser_score",
            "admission_type",
            "admission_location",
            "discharge_location",
            "insurance",
            "language",
            "race_group",
            "ethnicity_from_race",
            "marital_status",
            "hospital_los_days",
            "ed_los_hours",
            "hospital_expire_flag",
            "died_in_hospital",
            "has_deathtime",
            "has_dod",
            "n_labevents_rows",
            "n_microbiologyevents_rows",
            "n_poe_rows",
            "n_poe_detail_rows",
        ]
        if column in analysis.columns
    ]
    return analysis.loc[:, keep_columns].sort_values(
        [GROUP_COLUMN, "cohort", "pair_id", "subject_id", "hadm_id"]
    )


def summarize_numeric(analysis: pd.DataFrame) -> pd.DataFrame:
    """Summarize numeric outcomes/utilization by subgroup and cohort."""
    available_columns = [
        column for column in NUMERIC_SUMMARY_COLUMNS if column in analysis.columns
    ]
    rows = []
    for (group_name, group_label, cohort), group in analysis.groupby(
        [GROUP_COLUMN, "exclusive_combined_group_label", "cohort"],
        dropna=False,
    ):
        for column in available_columns:
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            rows.append(
                {
                    GROUP_COLUMN: group_name,
                    "exclusive_combined_group_label": group_label,
                    "cohort": cohort,
                    "measure": column,
                    "n_admissions": len(group),
                    "n_nonmissing": len(values),
                    "mean": values.mean() if len(values) else pd.NA,
                    "sd": values.std(ddof=1) if len(values) > 1 else pd.NA,
                    "median": values.median() if len(values) else pd.NA,
                    "q1": values.quantile(0.25) if len(values) else pd.NA,
                    "q3": values.quantile(0.75) if len(values) else pd.NA,
                    "iqr": (
                        values.quantile(0.75) - values.quantile(0.25)
                        if len(values)
                        else pd.NA
                    ),
                    "min": values.min() if len(values) else pd.NA,
                    "max": values.max() if len(values) else pd.NA,
                    "n_with_any_positive_value": int(values.gt(0).sum())
                    if len(values)
                    else 0,
                    "pct_with_any_positive_value": 100.0 * values.gt(0).mean()
                    if len(values)
                    else pd.NA,
                }
            )
    return pd.DataFrame(rows).sort_values([GROUP_COLUMN, "measure", "cohort"])


def summarize_binary(analysis: pd.DataFrame) -> pd.DataFrame:
    """Summarize binary mortality/death indicators by subgroup and cohort."""
    binary_columns = [
        column
        for column in ["died_in_hospital", "has_deathtime", "has_dod"]
        if column in analysis.columns
    ]
    rows = []
    for (group_name, group_label, cohort), group in analysis.groupby(
        [GROUP_COLUMN, "exclusive_combined_group_label", "cohort"],
        dropna=False,
    ):
        for column in binary_columns:
            values = group[column].dropna().astype(bool)
            rows.append(
                {
                    GROUP_COLUMN: group_name,
                    "exclusive_combined_group_label": group_label,
                    "cohort": cohort,
                    "measure": column,
                    "n_admissions": len(group),
                    "n_nonmissing": len(values),
                    "n_positive": int(values.sum()),
                    "pct_positive": 100.0 * values.mean() if len(values) else pd.NA,
                }
            )
    return pd.DataFrame(rows).sort_values([GROUP_COLUMN, "measure", "cohort"])


def summarize_categorical(analysis: pd.DataFrame) -> pd.DataFrame:
    """Summarize categorical descriptors by subgroup and cohort."""
    available_columns = [
        column for column in CATEGORICAL_COLUMNS if column in analysis.columns
    ]
    rows = []
    denominators = (
        analysis.groupby([GROUP_COLUMN, "exclusive_combined_group_label", "cohort"])[
            "hadm_id"
        ]
        .nunique()
        .to_dict()
    )
    for column in available_columns:
        values = analysis.loc[
            :, [GROUP_COLUMN, "exclusive_combined_group_label", "cohort", "hadm_id", column]
        ].copy()
        values[column] = values[column].fillna("missing").astype(str).str.strip()
        values.loc[values[column].eq(""), column] = "missing"
        counts = (
            values.groupby(
                [GROUP_COLUMN, "exclusive_combined_group_label", "cohort", column],
                as_index=False,
            )["hadm_id"]
            .nunique()
            .rename(columns={column: "category", "hadm_id": "n_admissions"})
        )
        counts["measure"] = column
        counts["pct_within_subgroup_cohort"] = counts.apply(
            lambda row: 100.0
            * row["n_admissions"]
            / denominators.get(
                (
                    row[GROUP_COLUMN],
                    row["exclusive_combined_group_label"],
                    row["cohort"],
                ),
                0,
            ),
            axis=1,
        )
        rows.append(counts)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).loc[
        :,
        [
            GROUP_COLUMN,
            "exclusive_combined_group_label",
            "cohort",
            "measure",
            "category",
            "n_admissions",
            "pct_within_subgroup_cohort",
        ],
    ].sort_values(
        [GROUP_COLUMN, "measure", "cohort", "n_admissions"],
        ascending=[True, True, True, False],
    )


def summarize_counts(analysis: pd.DataFrame) -> pd.DataFrame:
    """Count admissions and subjects in each chief-complaint subgroup/cohort cell."""
    return (
        analysis.groupby([GROUP_COLUMN, "exclusive_combined_group_label", "cohort"])
        .agg(
            n_admissions=("hadm_id", "nunique"),
            n_subjects=("subject_id", "nunique"),
            n_pairs=("pair_id", "nunique"),
        )
        .reset_index()
        .sort_values([GROUP_COLUMN, "cohort"])
    )


def build_pair_membership_summary(analysis: pd.DataFrame) -> pd.DataFrame:
    """Report how often both admissions from a pair fall into each subgroup."""
    rows = []
    for group_name, group in analysis.groupby(GROUP_COLUMN):
        group_label = GROUP_LABELS.get(group_name, group_name)
        pair_counts = group.groupby("pair_id")["cohort"].nunique()
        rows.append(
            {
                GROUP_COLUMN: group_name,
                "exclusive_combined_group_label": group_label,
                "n_pairs_with_any_group_admission": int(pair_counts.size),
                "n_pairs_with_both_cohorts_in_group": int(pair_counts.eq(2).sum()),
                "pct_pairs_with_both_cohorts_in_group": 100.0
                * pair_counts.eq(2).mean(),
            }
        )
    return pd.DataFrame(rows).sort_values(GROUP_COLUMN)



# Current paired pure-subgroup analysis. Existing helper functions remain
# available to the optional all-pure-admission exploration script.
PAIRED_INPUT_DIR = SCRIPT_DIR / "analysis_output_complete_top_five_cc_pairs"
PAIRED_WORKUP_OUTPUT_DIR = SCRIPT_DIR / "analysis_output_complete_cc_pair_workup"
WORKUP_COUNT_COLUMNS = ["n_labevents_rows", "n_microbiologyevents_rows", "n_poe_rows", "n_poe_detail_rows"]
WORKUP_MEASURES = ["hospital_los_days", "ed_los_hours", *WORKUP_COUNT_COLUMNS]

# Keep inference with the existing analysis instead of adding a parallel script.
# R's glmmTMB fits crossed NB2 models; no dependency downloads happen at runtime.
NEGATIVE_BINOMIAL_R = r'''
args <- commandArgs(trailingOnly=TRUE)
.libPaths(c(args[3], .libPaths()))
suppressPackageStartupMessages(library(glmmTMB))
data <- read.csv(args[1], check.names=FALSE)
out <- args[2]
dir.create(out, recursive=TRUE, showWarnings=FALSE)
data$mhc1 <- as.integer(data$cohort == "MHC1_psychotic")
data$patient <- factor(data$subject_id)
data$matched_pair <- factor(data$pair_id)
outcomes <- c("n_labevents_rows", "n_microbiologyevents_rows", "n_poe_rows", "n_poe_detail_rows")
results <- list(); diagnostics <- list(); coefficients <- list()
set.seed(20261005)
for (group in sort(unique(data$pure_cc_group))) {
  d <- droplevels(data[data$pure_cc_group == group, ])
  stopifnot(all(table(d$matched_pair) == 2), all(tapply(d$mhc1, d$matched_pair, sum) == 1))
  for (outcome in outcomes) {
    cat("Fitting NB2:", group, outcome, "\n"); flush.console()
    y <- d[[outcome]]
    stopifnot(!anyNA(y), all(y >= 0), all(y == floor(y)))
    warnings <- character()
    fit <- tryCatch(withCallingHandlers({
      formula <- as.formula(paste(outcome, "~ mhc1 + (1|patient) + (1|matched_pair)"))
      glmmTMB(formula, data=d, family=nbinom2(link="log"),
        control=glmmTMBControl(optCtrl=list(iter.max=10000, eval.max=10000)))
    },
      warning=function(w) { warnings <<- c(warnings, conditionMessage(w)); invokeRestart("muffleWarning") }),
      error=function(e) e)
    key <- paste(group, outcome, sep="__")
    family <- if (outcome == "n_poe_detail_rows") "order_detail_metadata" else "recorded_workup_counts"
    if (inherits(fit, "error")) {
      diagnostics[[key]] <- data.frame(cc_group=group, outcome=outcome, status="fit_failed",
        message=conditionMessage(fit), n_admissions=nrow(d), n_pairs=nlevels(d$matched_pair))
      next
    }
    conv <- if (fit$fit$convergence != 0) fit$fit$message else character()
    converged <- fit$fit$convergence == 0 && isTRUE(fit$sdr$pdHess)
    vc <- VarCorr(fit)$cond
    singular <- any(vapply(vc, function(x) as.numeric(x[1,1]) < 1e-8, logical(1)))
    cf <- summary(fit)$coefficients$cond
    beta <- cf["mhc1", "Estimate"]; se <- cf["mhc1", "Std. Error"]
    results[[key]] <- data.frame(cc_group=group, outcome=outcome, model="M0", analysis_family=family,
      n_admissions=nrow(d), n_pairs=nlevels(d$matched_pair), n_patients=nlevels(d$patient),
      beta_log_count=beta, standard_error=se, count_ratio=exp(beta),
      ci_low=exp(beta-1.96*se), ci_high=exp(beta+1.96*se), p_value=cf["mhc1", "Pr(>|z|)"],
      converged=converged, singular=singular)
    coefficients[[key]] <- data.frame(cc_group=group, outcome=outcome, term=rownames(cf),
      beta=cf[,1], standard_error=cf[,2], z=cf[,3], p_value=cf[,4], row.names=NULL)
    theta <- sigma(fit)
    mu <- fitted(fit)
    # Unconditional simulations redraw patient/pair effects. Compare marginal
    # variance and zero frequency as screening checks, not formal residual tests.
    # A raw conditional Pearson statistic is descriptive only: estimated random
    # effects absorb variability, so comparing it to 1 is not a calibrated test.
    sims <- as.matrix(simulate(fit, nsim=200))
    variance <- mu + mu^2/theta
    observed_dispersion <- mean((y-mu)^2/variance)
    simulated_variance <- apply(sims, 2, var)
    simulated_zeros <- colMeans(sims == 0)
    diagnostics[[key]] <- data.frame(cc_group=group, outcome=outcome,
      status=if (converged) "converged" else "convergence_warning",
      message=paste(unique(c(warnings, conv)), collapse=" | "),
      n_admissions=nrow(d), n_pairs=nlevels(d$matched_pair), singular=singular,
      nb_theta=theta, patient_variance=as.numeric(vc$patient[1,1]),
      pair_variance=as.numeric(vc$matched_pair[1,1]), AIC=AIC(fit),
      observed_zero_fraction=mean(y == 0),
      simulated_zero_low=quantile(simulated_zeros,.025), simulated_zero_high=quantile(simulated_zeros,.975),
      observed_conditional_pearson_statistic=observed_dispersion,
      observed_count_variance=var(y),
      simulated_count_variance_low=quantile(simulated_variance,.025),
      simulated_count_variance_high=quantile(simulated_variance,.975))
    saveRDS(fit, file.path(out, paste0(gsub("[^a-zA-Z0-9_]", "_", key), ".rds")))
  }
}
# Align differing diagnostic columns when a model fails.
bind_rows <- function(rows) {
  cols <- unique(unlist(lapply(rows, names)))
  do.call(rbind, lapply(rows, function(x) { x[setdiff(cols, names(x))] <- NA; x[cols] }))
}
if (length(results)) {
  r <- bind_rows(results); r$FDR <- NA_real_
  for (family in unique(r$analysis_family)) {
    ix <- which(r$analysis_family == family & r$converged)
    r$FDR[ix] <- p.adjust(r$p_value[ix], method="BH", n=if (family == "recorded_workup_counts") 15 else 5)
  }
  write.csv(r, file.path(out, "negative_binomial_cohort_effects.csv"), row.names=FALSE)
  write.csv(bind_rows(coefficients), file.path(out, "negative_binomial_coefficients.csv"), row.names=FALSE)
  print(r[,c("cc_group","outcome","count_ratio","ci_low","ci_high","FDR","converged","singular")], row.names=FALSE)
}
diag <- bind_rows(diagnostics)
diag$zero_screen_flag <- with(diag, observed_zero_fraction < simulated_zero_low | observed_zero_fraction > simulated_zero_high)
diag$variance_screen_flag <- with(diag, observed_count_variance < simulated_count_variance_low | observed_count_variance > simulated_count_variance_high)
write.csv(diag, file.path(out, "negative_binomial_diagnostics.csv"), row.names=FALSE)
writeLines(c(
  "M0: count ~ MHC1 + (1|patient) + (1|matched_pair). NB2 log link, ML with Laplace approximation.",
  "Complete original pairs within each pure CC group. MHC0 reference. No adjustment covariates or LOS offset.",
  "Ratios compare conditional expected recorded counts per admission, NOT rates per day or distinct tests.",
  "95% intervals and p values use asymptotic Wald inference; singular/convergence results must be reviewed.",
  "BH across 15 main comparisons (three outcomes x five CC groups); five metadata comparisons corrected separately.",
  "Failed/nonconverged models have no FDR. Unconditional simulation checks use 200 draws, seed 20261005.",
  "Zero frequency and marginal count variance are screening checks, not formal goodness-of-fit tests.",
  "POE includes non-diagnostic orders; POE detail is metadata. No claim of diagnostic overshadowing follows directly."
), file.path(out, "negative_binomial_methods.txt"))
capture.output(sessionInfo(), file=file.path(out, "R_session_info.txt"))
if (length(results) != length(unique(data$pure_cc_group))*length(outcomes)) stop("Some models failed; inspect diagnostics.")
'''


def fit_negative_binomial_models():
    """Fit NB2 crossed-intercept M0 models without creating another script."""
    dataset = PAIRED_WORKUP_OUTPUT_DIR / "complete_cc_pair_workup_dataset.csv"
    if not dataset.exists():
        raise FileNotFoundError(f"Run the descriptive analysis first: {dataset}")
    r_library = PROJECT_DIR / "03_discharge_note_text_analysis" / "01_language_complexity" / ".r-library"
    subprocess.run([
        "Rscript", "-e", NEGATIVE_BINOMIAL_R, str(dataset),
        str(PAIRED_WORKUP_OUTPUT_DIR / "negative_binomial_models"), str(r_library),
    ], check=True)
    model_dir = PAIRED_WORKUP_OUTPUT_DIR / "negative_binomial_models"
    (model_dir / "model_input_sha256.txt").write_text(hashlib.sha256(dataset.read_bytes()).hexdigest())
    (model_dir / "model_input_status.txt").write_text("Model estimates match the current work-up dataset.\n")


LONGITUDINAL_R = r'''
args <- commandArgs(trailingOnly=TRUE)
.libPaths(c(args[3], .libPaths()))
suppressPackageStartupMessages(library(glmmTMB))
data <- read.csv(args[1], check.names=FALSE)
out <- args[2]
dir.create(out, recursive=TRUE, showWarnings=FALSE)
stopifnot(!anyNA(data$exposure_days), all(data$exposure_days >= 0),
          all(data$n_events[data$exposure_days == 0] == 0))
data <- data[data$exposure_days > 0, ]
data$mhc1 <- as.integer(data$cohort == "MHC1_psychotic")
data$patient <- factor(data$subject_id)
data$matched_pair <- factor(data$pair_id)
data$admission <- factor(data$hadm_id)
data$period <- factor(data$interval, levels=c("0–24h", "24–48h", "48–72h", "days 4–7", "after day 7"))
stopifnot(!anyNA(data$period), !anyNA(data$n_events), all(data$n_events >= 0),
          all(data$n_events == floor(data$n_events)), length(unique(data$pure_cc_group)) == 5)
measures <- c("laboratory_measurements", "laboratory_specimens", "microbiology_records", "provider_order_activity")
stopifnot(setequal(unique(data$measure), measures))
effects <- list(); diagnostics <- list(); coefficients <- list(); risks <- list()
set.seed(20261006)
for (group in sort(unique(data$pure_cc_group))) for (measure in measures) {
  d <- droplevels(data[data$pure_cc_group == group & data$measure == measure, ])
  admissions <- unique(d[,c("admission", "matched_pair", "mhc1")])
  stopifnot(all(table(admissions$matched_pair) == 2),
            all(tapply(admissions$mhc1, admissions$matched_pair, sum) == 1),
            !anyDuplicated(d[,c("admission", "period")]))
  key <- paste(group, measure, sep="__")
  cat("Longitudinal NB2:", group, measure, "\n"); flush.console()
  warnings <- character()
  fit <- tryCatch(withCallingHandlers(
    glmmTMB(n_events ~ mhc1 + period + offset(log(exposure_days)) +
            (1|patient) + (1|matched_pair) + (1|admission),
      data=d, family=nbinom2(link="log"),
      control=glmmTMBControl(optCtrl=list(iter.max=10000, eval.max=10000))),
    warning=function(w) {warnings <<- c(warnings, conditionMessage(w)); invokeRestart("muffleWarning")}),
    error=function(e) e)
  if (inherits(fit, "error")) stop(key, ": ", conditionMessage(fit))
  cf <- summary(fit)$coefficients$cond
  vc <- VarCorr(fit)$cond
  converged <- fit$fit$convergence == 0 && isTRUE(fit$sdr$pdHess)
  singular <- any(vapply(vc, function(x) as.numeric(x[1,1]) < 1e-8, logical(1)))
  beta <- cf["mhc1",1]; se <- cf["mhc1",2]
  effects[[key]] <- data.frame(cc_group=group, outcome=measure,
    n_intervals=nrow(d), n_admissions=nlevels(d$admission), n_pairs=nlevels(d$matched_pair),
    n_patients=nlevels(d$patient), beta_log_rate=beta, standard_error=se,
    rate_ratio=exp(beta), ci_low=exp(beta-1.96*se), ci_high=exp(beta+1.96*se),
    p_value=cf["mhc1",4], converged=converged, singular=singular)
  coefficients[[key]] <- data.frame(cc_group=group, outcome=measure, term=rownames(cf),
    beta=cf[,1], standard_error=cf[,2], z=cf[,3], p_value=cf[,4], row.names=NULL)
  sims <- as.matrix(simulate(fit, nsim=200))
  zeros <- colMeans(sims == 0); variances <- apply(sims, 2, var)
  zero_limits <- quantile(zeros,c(.025,.975)); variance_limits <- quantile(variances,c(.025,.975))
  # Descriptive adjacent-interval residual correlation: not a calibrated test.
  d$residual <- residuals(fit, type="pearson")
  ordered <- d[order(d$admission, d$period), ]
  same <- head(ordered$admission,-1) == tail(ordered$admission,-1)
  adjacent <- cor(head(ordered$residual,-1)[same], tail(ordered$residual,-1)[same])
  diagnostics[[key]] <- data.frame(cc_group=group, outcome=measure, converged=converged,
    positive_hessian=isTRUE(fit$sdr$pdHess), singular=singular,
    optimizer_message=paste(fit$fit$message,collapse=" | "), warnings=paste(unique(warnings),collapse=" | "),
    patient_variance=as.numeric(vc$patient[1,1]), pair_variance=as.numeric(vc$matched_pair[1,1]),
    admission_variance=as.numeric(vc$admission[1,1]), nb_theta=sigma(fit), AIC=AIC(fit),
    observed_zero_fraction=mean(d$n_events == 0), simulated_zero_low=zero_limits[1], simulated_zero_high=zero_limits[2],
    zero_screen_flag=mean(d$n_events == 0) < zero_limits[1] | mean(d$n_events == 0) > zero_limits[2],
    observed_count_variance=var(d$n_events), simulated_variance_low=variance_limits[1], simulated_variance_high=variance_limits[2],
    variance_screen_flag=var(d$n_events) < variance_limits[1] | var(d$n_events) > variance_limits[2],
    adjacent_conditional_pearson_correlation=adjacent)
  risks[[key]] <- aggregate(cbind(n_events, exposure_days) ~ cohort + period, d, sum)
  risks[[key]]$cc_group <- group; risks[[key]]$outcome <- measure
  saveRDS(fit,file.path(out,paste0(gsub("[^a-zA-Z0-9_]","_",key),".rds")))
}
r <- do.call(rbind,effects); r$FDR <- NA_real_
valid <- which(r$converged & is.finite(r$p_value))
r$FDR[valid] <- p.adjust(r$p_value[valid],method="BH",n=20)
write.csv(r,file.path(out,"longitudinal_cohort_effects.csv"),row.names=FALSE)
write.csv(do.call(rbind,coefficients),file.path(out,"longitudinal_coefficients.csv"),row.names=FALSE)
write.csv(do.call(rbind,diagnostics),file.path(out,"longitudinal_diagnostics.csv"),row.names=FALSE)
write.csv(do.call(rbind,risks),file.path(out,"longitudinal_exposure_summary.csv"),row.names=FALSE)
writeLines(c(
 "Count ~ cohort + categorical period + offset(log(observed days)) + patient, pair and admission random intercepts.",
 "NB2 log link; maximum likelihood with Laplace approximation. MHC0 and first 24h are references.",
 "One constant conditional MHC1/MHC0 rate ratio across intervals; no cohort-by-time interaction or day-specific tests.",
 "Four outcomes x five CCs. BH FDR across all 20 planned cohort effects, including specimens.",
 "95% Wald confidence intervals and asymptotic Wald p values. Review convergence and boundary fits before interpretation.",
 "Only positive exposure intervals; partial periods use actual inpatient time; no post-discharge zero imputation.",
 "Timing cohort preserves 1990 original pairs, excluding 3 pairs with invalid/nonpositive duration. Extremes retained.",
 "Corrected laboratory linkage included; microbiology and POE retain original admission links. POE detail excluded.",
 "Specimens counted once at earliest timestamp, not assumed equivalent to blood draws or tests ordered.",
 "Later periods compare patients still hospitalized; no correction for informative discharge or causal inference.",
 "Random intercepts do not assume autoregressive temporal correlation; adjacent Pearson correlation is descriptive only.",
 "Unconditional simulations (200, seed 20261006) screen zero frequency/count variance; not formal fit tests."
),file.path(out,"longitudinal_methods.txt"))
capture.output(sessionInfo(),file=file.path(out,"R_session_info.txt"))
print(r[,c("cc_group","outcome","rate_ratio","ci_low","ci_high","FDR","converged","singular")],row.names=FALSE)
'''


def fit_longitudinal_models():
    """Use saved inpatient intervals only; no database/raw text access."""
    dataset = SCRIPT_DIR / "analysis_output_lab_linkage_and_timing" / "clinical_event_admission_intervals.csv"
    if not dataset.exists():
        raise FileNotFoundError(f"Run 06 descriptive timing first: {dataset}")
    output = PAIRED_WORKUP_OUTPUT_DIR / "longitudinal_negative_binomial_models"
    r_library = PROJECT_DIR / "03_discharge_note_text_analysis" / "01_language_complexity" / ".r-library"
    fingerprint = hashlib.sha256(dataset.read_bytes()).hexdigest()
    subprocess.run(["Rscript", "-e", LONGITUDINAL_R, str(dataset), str(output), str(r_library)], check=True)
    if hashlib.sha256(dataset.read_bytes()).hexdigest() != fingerprint:
        raise RuntimeError("Timing inputs changed while models were fitting")
    (output / "model_input_sha256.txt").write_text(fingerprint + "\n")


LONGITUDINAL_ZI_R = r'''
args <- commandArgs(trailingOnly=TRUE)
.libPaths(c(args[4], .libPaths()))
suppressPackageStartupMessages(library(glmmTMB))
data <- read.csv(args[1],check.names=FALSE)
model_dir <- args[2]; out <- args[3]
dir.create(out,recursive=TRUE,showWarnings=FALSE)
data <- data[data$exposure_days > 0,]
data$mhc1 <- as.integer(data$cohort == "MHC1_psychotic")
data$patient <- factor(data$subject_id); data$matched_pair <- factor(data$pair_id)
data$admission <- factor(data$hadm_id)
data$period <- factor(data$interval,levels=c("0–24h","24–48h","48–72h","days 4–7","after day 7"))
stopifnot(!anyNA(data$period))
effects <- list(); diagnostics <- list(); coefficients <- list(); strata <- list()
set.seed(20261008)
band <- function(x) as.numeric(quantile(x,c(.025,.975)))
for (group in sort(unique(data$pure_cc_group))) for (measure in c("laboratory_measurements","laboratory_specimens")) {
  key <- paste(group,measure,sep="__")
  cat("Longitudinal ZINB2:",key,"\n"); flush.console()
  d <- droplevels(data[data$pure_cc_group == group & data$measure == measure,])
  original <- readRDS(file.path(model_dir,paste0(gsub("[^a-zA-Z0-9_]","_",key),".rds")))
  frame <- model.frame(original)
  stopifnot(identical(as.numeric(frame$n_events),as.numeric(d$n_events)),
    identical(as.character(frame$admission),as.character(d$admission)),
    identical(as.character(frame$period),as.character(d$period)),
    isTRUE(all.equal(as.numeric(model.offset(frame)),log(d$exposure_days))))
  warnings <- character()
  fit <- tryCatch(withCallingHandlers(
    glmmTMB(n_events ~ mhc1 + period + offset(log(exposure_days)) +
      (1|patient) + (1|matched_pair) + (1|admission), ziformula=~1,
      data=d,family=nbinom2(link="log"),
      control=glmmTMBControl(optCtrl=list(iter.max=10000,eval.max=10000))),
    warning=function(w) {warnings <<- c(warnings,conditionMessage(w)); invokeRestart("muffleWarning")}),
    error=function(e) e)
  if (inherits(fit,"error")) {
    diagnostics[[key]] <- data.frame(cc_group=group,outcome=measure,converged=FALSE,
      status="fit_failed",warnings=conditionMessage(fit))
    next
  }
  converged <- fit$fit$convergence==0 && isTRUE(fit$sdr$pdHess)
  cf <- summary(fit)$coefficients$cond
  zcf <- summary(fit)$coefficients$zi
  beta <- cf["mhc1",1]; se <- cf["mhc1",2]
  zi_beta <- zcf[1,1]; zi_se <- zcf[1,2]; pi <- plogis(zi_beta)
  vc <- VarCorr(fit)$cond
  singular <- any(vapply(vc,function(x) as.numeric(x[1,1])<1e-8,logical(1)))
  effects[[key]] <- data.frame(cc_group=group,outcome=measure,model="ZINB2_constant_zero_probability",
    n_intervals=nrow(d),n_admissions=nlevels(d$admission),n_pairs=nlevels(d$matched_pair),
    n_patients=nlevels(d$patient),beta_log_rate=beta,standard_error=se,rate_ratio=exp(beta),
    ci_low=exp(beta-1.96*se),ci_high=exp(beta+1.96*se),p_value=cf["mhc1",4],
    extra_zero_probability=pi,extra_zero_ci_low=plogis(zi_beta-1.96*zi_se),
    extra_zero_ci_high=plogis(zi_beta+1.96*zi_se),converged=converged,singular=singular)
  for (component in c("cond","zi")) {
    tab <- summary(fit)$coefficients[[component]]
    coefficients[[paste(key,component)]] <- data.frame(cc_group=group,outcome=measure,
      component=component,term=rownames(tab),beta=tab[,1],standard_error=tab[,2],z=tab[,3],p_value=tab[,4],row.names=NULL)
  }
  sims <- as.matrix(simulate(fit,nsim=500))
  y <- d$n_events; mu <- as.numeric(predict(fit,type="conditional")); theta <- sigma(fit)
  expected_zero <- pi+(1-pi)*dnbinom(0,mu=mu,size=theta)
  zb <- band(colMeans(sims==0)); vb <- band(apply(sims,2,var))
  pearson <- as.numeric(residuals(fit,type="pearson"))
  ord <- order(d$admission,d$period)
  same <- head(d$admission[ord],-1)==tail(d$admission[ord],-1)
  previous <- head(pearson[ord],-1)[same]; following <- tail(pearson[ord],-1)[same]
  diagnostics[[key]] <- data.frame(cc_group=group,outcome=measure,converged=converged,
    status=if(converged) "converged" else "convergence_warning",positive_hessian=isTRUE(fit$sdr$pdHess),
    singular=singular,optimizer_message=paste(fit$fit$message,collapse=" | "),warnings=paste(unique(warnings),collapse=" | "),
    patient_variance=as.numeric(vc$patient[1,1]),pair_variance=as.numeric(vc$matched_pair[1,1]),
    admission_variance=as.numeric(vc$admission[1,1]),nb_theta=theta,extra_zero_probability=pi,
    original_AIC=AIC(original),ZI_AIC=AIC(fit),AIC_change=AIC(fit)-AIC(original),
    observed_zero_fraction=mean(y==0),conditional_expected_zero_fraction=mean(expected_zero),
    simulated_zero_low=zb[1],simulated_zero_high=zb[2],zero_screen_flag=mean(y==0)<zb[1] | mean(y==0)>zb[2],
    observed_variance=var(y),simulated_variance_low=vb[1],simulated_variance_high=vb[2],
    variance_screen_flag=var(y)<vb[1] | var(y)>vb[2],
    adjacent_pearson_correlation=cor(previous,following),
    adjacent_pearson_correlation_bounded=cor(pmax(-3,pmin(3,previous)),pmax(-3,pmin(3,following))))
  for (cohort in unique(d$cohort)) for (period in levels(d$period)) {
    ix <- which(d$cohort==cohort & d$period==period)
    if (!length(ix)) next
    z <- band(colMeans(sims[ix,,drop=FALSE]==0))
    strata[[paste(key,cohort,period)]] <- data.frame(cc_group=group,outcome=measure,
      cohort=cohort,interval=period,n_intervals=length(ix),observed_zero_fraction=mean(y[ix]==0),
      conditional_expected_zero_fraction=mean(expected_zero[ix]),simulated_zero_low=z[1],simulated_zero_high=z[2],
      mean_pearson_residual=mean(pearson[ix]),pearson_mean_square=mean(pearson[ix]^2))
  }
  saveRDS(fit,file.path(out,paste0(gsub("[^a-zA-Z0-9_]","_",key),".rds")))
}
bind_rows <- function(rows) {
  cols <- unique(unlist(lapply(rows,names)))
  do.call(rbind,lapply(rows,function(x) {x[setdiff(cols,names(x))] <- NA; x[cols]}))
}
diag <- bind_rows(diagnostics)
write.csv(diag,file.path(out,"zero_inflated_diagnostics.csv"),row.names=FALSE)
if (length(effects)) {
  r <- bind_rows(effects)
  original_effects <- read.csv(file.path(model_dir,"longitudinal_cohort_effects.csv"))
  # Keep the original 20-comparison clinical family: ten ZI lab estimates plus
  # ten unchanged NB microbiology/POE estimates. This is exploratory, not a
  # replacement of the primary results and not selection by significant p.
  unchanged <- original_effects[!original_effects$outcome %in% c("laboratory_measurements","laboratory_specimens"),]
  unchanged$model <- "Original_NB2_nonlaboratory"
  candidate <- bind_rows(list(r,unchanged)); candidate$FDR <- NA_real_
  valid <- which(candidate$converged & is.finite(candidate$p_value))
  candidate$FDR[valid] <- p.adjust(candidate$p_value[valid],method="BH",n=20)
  r <- merge(r[,setdiff(names(r),"FDR")],candidate[,c("cc_group","outcome","FDR")],by=c("cc_group","outcome"),sort=FALSE)
  write.csv(r,file.path(out,"zero_inflated_cohort_effects.csv"),row.names=FALSE)
  write.csv(candidate,file.path(out,"exploratory_20_outcome_family.csv"),row.names=FALSE)
  write.csv(bind_rows(coefficients),file.path(out,"zero_inflated_coefficients.csv"),row.names=FALSE)
  comparison <- merge(original_effects[original_effects$outcome %in% r$outcome,],r,
    by=c("cc_group","outcome"),suffixes=c("_NB2","_ZI"))
  stopifnot(all(comparison$n_intervals_NB2==comparison$n_intervals_ZI),
    all(comparison$n_admissions_NB2==comparison$n_admissions_ZI),all(comparison$n_pairs_NB2==comparison$n_pairs_ZI))
  comparison <- merge(comparison,diag[,c("cc_group","outcome","AIC_change","zero_screen_flag","variance_screen_flag")],by=c("cc_group","outcome"))
  write.csv(comparison,file.path(out,"NB2_ZI_comparison.csv"),row.names=FALSE)
  write.csv(bind_rows(strata),file.path(out,"zero_inflated_by_cohort_interval.csv"),row.names=FALSE)
  print(r[,c("cc_group","outcome","rate_ratio","ci_low","ci_high","FDR","extra_zero_probability","converged","singular")],row.names=FALSE)
}
writeLines(c(
  "Exploratory ZINB2 alternative for both laboratory outcomes in all five CCs; primary outputs unchanged.",
  "Count model identical to NB2: cohort + categorical period + log exposure offset + patient/pair/admission intercepts.",
  "Zero model: logit(pi) = intercept only; extra-zero probability common to cohorts, periods and random-effect levels.",
  "Expected count=(1-pi)*mu; exp(cohort beta) is also the overall expected-rate ratio because pi is common.",
  "Same admissions/intervals, no outlier exclusions, partial exposure unchanged, no new raw-data access.",
  "ML/Laplace, 95% Wald intervals; convergence/boundaries and diagnostic checks required before interpreting inference.",
  "Exploratory BH family remains 20: ten ZI laboratory plus ten unchanged original nonlaboratory effects.",
  "Do not select the primary model based on significance. AIC compared only on identical outcome and observations.",
  "No naive chi-square likelihood-ratio test: zero inflation is on a parameter boundary under the NB null.",
  "500 unconditional simulation draws, seed 20261008; screens are not formal goodness-of-fit tests.",
  "An extra-zero component does not identify why observations are zero; no structural-zero patient classification.",
  "Does not by itself fix informative discharge, varying group effects over time, or residual temporal dependence."
),file.path(out,"zero_inflated_methods.txt"))
capture.output(sessionInfo(),file=file.path(out,"R_session_info.txt"))
'''


def fit_zero_inflated_longitudinal_models():
    """Fit the agreed constant-zero laboratory alternative; preserve NB fits."""
    dataset = SCRIPT_DIR / "analysis_output_lab_linkage_and_timing" / "clinical_event_admission_intervals.csv"
    model_dir = PAIRED_WORKUP_OUTPUT_DIR / "longitudinal_negative_binomial_models"
    fingerprint = hashlib.sha256(dataset.read_bytes()).hexdigest()
    saved = model_dir / "model_input_sha256.txt"
    if not saved.exists() or saved.read_text().strip() != fingerprint:
        raise ValueError("Original longitudinal models do not match current interval inputs")
    output = model_dir / "zero_inflated_laboratory_models"
    r_library = PROJECT_DIR / "03_discharge_note_text_analysis" / "01_language_complexity" / ".r-library"
    subprocess.run(["Rscript", "-", str(dataset), str(model_dir), str(output), str(r_library)],
                   input=LONGITUDINAL_ZI_R, text=True, check=True)
    if hashlib.sha256(dataset.read_bytes()).hexdigest() != fingerprint:
        raise RuntimeError("Interval inputs changed during the zero-inflated model run")
    if not (output / "zero_inflated_diagnostics.csv").exists():
        raise RuntimeError("Zero-inflated diagnostics were not produced")
    (output / "model_input_sha256.txt").write_text(fingerprint + "\n")


ADJUSTED_LONGITUDINAL_R = r'''
args <- commandArgs(trailingOnly=TRUE)
.libPaths(c(args[3],.libPaths()))
ordinary_lab <- length(args)>=4 && args[4]=="M2_ordinary_laboratory"
without_admission <- length(args)>=4 && args[4]=="no_admission_M0_M2"
suppressPackageStartupMessages(library(glmmTMB))
data <- read.csv(args[1],check.names=FALSE); out <- args[2]
dir.create(out,recursive=TRUE,showWarnings=FALSE)
data <- data[data$exposure_days > 0,]
data$mhc1 <- as.integer(data$cohort=="MHC1_psychotic")
data$patient <- factor(data$subject_id); data$matched_pair <- factor(data$pair_id)
data$admission <- factor(data$hadm_id)
data$period <- factor(data$interval,levels=c("0–24h","24–48h","48–72h","days 4–7","after day 7"))
data$language_group <- factor(data$language_group,levels=c("English","Non-English","Missing"))
data$race_ethnicity_group <- factor(data$race_ethnicity_group,
 levels=c("White","Black","Asian","Hispanic/Latino","Other recorded categories","Unknown/declined/missing"))
stopifnot(!anyNA(data))
models <- list(
 M1_age_elixhauser=c("age_at_admission_per_10y","elixhauser_score_per_5pt"),
 M2_plus_prior_utilization=c("age_at_admission_per_10y","elixhauser_score_per_5pt","log1p_prior_all_admissions"),
 M3_plus_language=c("age_at_admission_per_10y","elixhauser_score_per_5pt","log1p_prior_all_admissions","language_group"),
 M4_plus_race_ethnicity=c("age_at_admission_per_10y","elixhauser_score_per_5pt","log1p_prior_all_admissions","language_group","race_ethnicity_group"))
if(ordinary_lab) {
 models <- models["M2_plus_prior_utilization"]
 data <- data[data$measure %in% c("laboratory_measurements","laboratory_specimens"),]
}
if(without_admission)models <- list(M0_time_only=character(),M2_plus_prior_utilization=models$M2_plus_prior_utilization)
effects <- list(); diagnostics <- list(); coefficients <- list(); strata <- list()
set.seed(20261009)
band <- function(x) as.numeric(quantile(x,c(.025,.975)))
for (model in names(models)) for (group in sort(unique(data$pure_cc_group))) for (measure in sort(unique(data$measure))) {
 key <- paste(model,group,measure,sep="__")
 cat("Adjusted longitudinal:",key,"\n"); flush.console()
 d <- droplevels(data[data$pure_cc_group==group & data$measure==measure,])
 admissions <- unique(d[,c("admission","matched_pair","mhc1")])
 stopifnot(all(table(admissions$matched_pair)==2),all(tapply(admissions$mhc1,admissions$matched_pair,sum)==1))
 is_lab <- measure %in% c("laboratory_measurements","laboratory_specimens") && !ordinary_lab
 terms <- c("mhc1","period",models[[model]],"offset(log(exposure_days))","(1|patient)","(1|matched_pair)")
 if(!without_admission)terms <- c(terms,"(1|admission)")
 form <- as.formula(paste("n_events ~",paste(terms,collapse=" + ")))
 warnings <- character()
 fit <- tryCatch(withCallingHandlers(
   glmmTMB(form,data=d,family=nbinom2(link="log"),ziformula=if(is_lab) ~1 else ~0,
     control=glmmTMBControl(optCtrl=list(iter.max=10000,eval.max=10000))),
   warning=function(w) {warnings <<- c(warnings,conditionMessage(w)); invokeRestart("muffleWarning")}),error=function(e)e)
 if (inherits(fit,"error")) {
   diagnostics[[key]] <- data.frame(model=model,cc_group=group,outcome=measure,converged=FALSE,status="fit_failed",warnings=conditionMessage(fit))
   next
 }
 converged <- fit$fit$convergence==0 && isTRUE(fit$sdr$pdHess)
 cf <- summary(fit)$coefficients$cond; vc <- VarCorr(fit)$cond
 beta <- cf["mhc1",1]; se <- cf["mhc1",2]
 pi <- if(is_lab) plogis(fixef(fit)$zi[1]) else 0
 singular <- any(vapply(vc,function(x)as.numeric(x[1,1])<1e-8,logical(1)))
 effects[[key]] <- data.frame(model=model,cc_group=group,outcome=measure,distribution=if(is_lab)"ZINB2" else "NB2",
  n_intervals=nrow(d),n_admissions=nlevels(d$admission),n_pairs=nlevels(d$matched_pair),n_patients=nlevels(d$patient),
  beta_log_rate=beta,standard_error=se,rate_ratio=exp(beta),ci_low=exp(beta-1.96*se),ci_high=exp(beta+1.96*se),
  p_value=cf["mhc1",4],extra_zero_probability=as.numeric(pi),converged=converged,singular=singular)
 for (component in if(is_lab)c("cond","zi") else "cond") {
   tab <- summary(fit)$coefficients[[component]]
   coefficients[[paste(key,component)]] <- data.frame(model=model,cc_group=group,outcome=measure,component=component,
    term=rownames(tab),beta=tab[,1],standard_error=tab[,2],z=tab[,3],p_value=tab[,4],row.names=NULL)
 }
 sims <- as.matrix(simulate(fit,nsim=500))
 y <- d$n_events; mu <- as.numeric(predict(fit,type="conditional")); theta <- sigma(fit)
 predicted_zero <- pi+(1-pi)*dnbinom(0,mu=mu,size=theta)
 zb <- band(colMeans(sims==0)); vb <- band(apply(sims,2,var))
 pearson <- as.numeric(residuals(fit,type="pearson"))
 ord <- order(d$admission,d$period)
 same <- head(d$admission[ord],-1)==tail(d$admission[ord],-1)
 previous <- head(pearson[ord],-1)[same]; following <- tail(pearson[ord],-1)[same]
 diagnostics[[key]] <- data.frame(model=model,cc_group=group,outcome=measure,converged=converged,
  status=if(converged)"converged" else "convergence_warning",positive_hessian=isTRUE(fit$sdr$pdHess),singular=singular,
  optimizer_message=paste(fit$fit$message,collapse=" | "),warnings=paste(unique(warnings),collapse=" | "),
  patient_variance=as.numeric(vc$patient[1,1]),pair_variance=as.numeric(vc$matched_pair[1,1]),
  admission_variance=if("admission" %in% names(vc))as.numeric(vc$admission[1,1]) else 0,
  extra_zero_probability=as.numeric(pi),nb_theta=theta,AIC=AIC(fit),
  observed_zero_fraction=mean(y==0),conditional_expected_zero_fraction=mean(predicted_zero),
  simulated_zero_low=zb[1],simulated_zero_high=zb[2],zero_screen_flag=mean(y==0)<zb[1] | mean(y==0)>zb[2],
  observed_variance=var(y),simulated_variance_low=vb[1],simulated_variance_high=vb[2],
  variance_screen_flag=var(y)<vb[1] | var(y)>vb[2],
  adjacent_pearson_correlation=cor(previous,following),
  adjacent_pearson_correlation_bounded=cor(pmax(-3,pmin(3,previous)),pmax(-3,pmin(3,following))))
 for (cohort in unique(d$cohort)) for (period in levels(d$period)) {
   ix <- which(d$cohort==cohort & d$period==period); if(!length(ix))next
   z <- band(colMeans(sims[ix,,drop=FALSE]==0))
   strata[[paste(key,cohort,period)]] <- data.frame(model=model,cc_group=group,outcome=measure,cohort=cohort,interval=period,
     n_intervals=length(ix),observed_zero_fraction=mean(y[ix]==0),conditional_expected_zero_fraction=mean(predicted_zero[ix]),
     simulated_zero_low=z[1],simulated_zero_high=z[2],mean_pearson_residual=mean(pearson[ix]),pearson_mean_square=mean(pearson[ix]^2))
 }
 saveRDS(fit,file.path(out,paste0(gsub("[^a-zA-Z0-9_]","_",key),".rds")))
}
bind_rows <- function(rows) {
 cols <- unique(unlist(lapply(rows,names)))
 do.call(rbind,lapply(rows,function(x){x[setdiff(cols,names(x))]<-NA;x[cols]}))
}
diag <- bind_rows(diagnostics)
write.csv(diag,file.path(out,"adjusted_longitudinal_diagnostics.csv"),row.names=FALSE)
if(length(effects)) {
 r <- bind_rows(effects); r$FDR <- NA_real_
 for(model in names(models)) {
  ix <- which(r$model==model & r$converged & is.finite(r$p_value))
  r$FDR[ix] <- p.adjust(r$p_value[ix],"BH",n=20)
 }
 write.csv(r,file.path(out,"adjusted_longitudinal_cohort_effects.csv"),row.names=FALSE)
 write.csv(bind_rows(coefficients),file.path(out,"adjusted_longitudinal_coefficients.csv"),row.names=FALSE)
 write.csv(bind_rows(strata),file.path(out,"adjusted_diagnostics_by_cohort_interval.csv"),row.names=FALSE)
 print(aggregate(cbind(zero_screen_flag,variance_screen_flag,singular)~model,diag,sum))
}
writeLines(c(
 "M1: cohort + time + age/10 + Elixhauser/5; M2 adds log1p(prior admissions); M3 adds language; M4 adds race/ethnicity.",
 "All use observed-days log offset and patient, pair, admission random intercepts; same complete timing pairs, extremes retained.",
 "Laboratory measurements/specimens: ZINB2 with constant extra-zero intercept. Microbiology/POE: NB2 without extra-zero component.",
 "No zero-component covariates or cohort-time interactions. Ratios assume constant relative cohort effects across intervals/covariates.",
 "English and White references; language/race unknown categories retained. These are measured covariate adjustments, not causal effects.",
 "ML/Laplace, asymptotic Wald p values and 95% intervals. BH across 20 cohort effects separately for each adjustment stage.",
 "Nonconverged estimates receive no FDR. Review convergence and remaining misfit before reporting significance.",
 "500 unconditional simulations per fit (seed 20261009); screening envelopes are exploratory, not formal fit tests.",
 "Observed zero proportions/variability unchanged across stages; simulations vary, including Monte Carlo fluctuation near boundaries.",
 "Residual correlations are descriptive; no independence-based test. Informative discharge remains unaddressed.",
 "Existing M0/NB/ZI and language-complexity results are not overwritten. M3/M4 are communication-context sensitivity extensions."
),file.path(out,"adjusted_longitudinal_methods.txt"))
capture.output(sessionInfo(),file=file.path(out,"R_session_info.txt"))
if(ordinary_lab)writeLines(c(
 "M2 ordinary NB2 comparison: laboratory measurements and specimens across five CCs.",
 "Same time, age/10, Elixhauser/5, log1p prior admissions, exposure offset and patient/pair/admission intercepts as existing ZI M2.",
 "No extra-zero component; no excluded observations. ML/Laplace, Wald inference, 500 diagnostic simulation draws.",
 "R FDR is provisional; Python reporting combines ten ordinary laboratory and ten unchanged nonlaboratory M2 effects into the 20-comparison family.",
 "Existing ZI M2 fits reused only after verifying input fingerprint. AIC comparisons require identical observations and valid fits.",
 "Not selected on cohort significance; no naive boundary likelihood-ratio test. Baseline M0 remains in reporting."
),file.path(out,"adjusted_longitudinal_methods.txt"))
if(without_admission)writeLines(c(
 "Consistent random-effects sensitivity: M0 and M2 across all five CCs and four event outcomes.",
 "Only admission intercept removed; patient and matched-pair intercepts retained.",
 "M0 includes cohort, categorical period and exposure offset; M2 adds age/10, Elixhauser/5 and log1p prior admissions.",
 "Laboratories retain constant-zero ZINB2; microbiology/POE retain NB2. All previous distributions unchanged.",
 "Identical admissions and intervals, no extreme exclusions. Same patients contribute within/between-admission observations.",
 "This assumes no additional admission-specific dependence beyond patient/pair intercepts; residual dependence remains a diagnostic concern.",
 "ML/Laplace, Wald inference, BH across twenty cohort comparisons separately in M0 and M2.",
 "500 unconditional simulation draws per model. Screens are exploratory; changed flags can reflect Monte Carlo variation.",
 "Admission variance shown as zero is fixed by removal, not an estimated boundary fit.",
 "Original full-intercept estimates remain intact; this sensitivity is not selected according to significance."
),file.path(out,"adjusted_longitudinal_methods.txt"))
'''


ADJUSTED_RETRY_R = r'''
args <- commandArgs(trailingOnly=TRUE)
.libPaths(c(args[3],.libPaths()))
suppressPackageStartupMessages(library(glmmTMB))
data <- read.csv(args[1]); base <- args[2]; out <- file.path(base,"optimizer_review")
dir.create(out,recursive=TRUE,showWarnings=FALSE)
diag <- read.csv(file.path(base,"adjusted_longitudinal_diagnostics.csv"))
targets <- diag[!diag$converged,]
data$mhc1 <- as.integer(data$cohort=="MHC1_psychotic")
data$patient <- factor(data$subject_id);data$matched_pair <- factor(data$pair_id);data$admission <- factor(data$hadm_id)
data$period <- factor(data$interval,levels=c("0–24h","24–48h","48–72h","days 4–7","after day 7"))
data$language_group <- factor(data$language_group,levels=c("English","Non-English","Missing"))
data$race_ethnicity_group <- factor(data$race_ethnicity_group,levels=c("White","Black","Asian","Hispanic/Latino","Other recorded categories","Unknown/declined/missing"))
rows <- list();set.seed(20261010)
for(i in seq_len(nrow(targets))) {
 target <- targets[i,];key <- paste(target$model,target$cc_group,target$outcome,sep="__")
 cat("Same-model BFGS retry:",key,"\n");flush.console()
 d <- droplevels(data[data$pure_cc_group==target$cc_group & data$measure==target$outcome,])
 old <- readRDS(file.path(base,paste0(gsub("[^a-zA-Z0-9_]","_",key),".rds")))
 is_lab <- target$outcome %in% c("laboratory_measurements","laboratory_specimens")
 warnings <- character()
 fit <- withCallingHandlers(glmmTMB(formula(old),data=d,family=nbinom2(link="log"),ziformula=if(is_lab)~1 else ~0,
   control=glmmTMBControl(optimizer=optim,optArgs=list(method="BFGS"),optCtrl=list(maxit=10000))),
   warning=function(w){warnings <<- c(warnings,conditionMessage(w));invokeRestart("muffleWarning")})
 cf <- summary(fit)$coefficients$cond;vc <- VarCorr(fit)$cond
 beta <- cf["mhc1",1];se <- cf["mhc1",2]
 converged <- fit$fit$convergence==0 && isTRUE(fit$sdr$pdHess)
 no_worse <- fit$fit$objective <= old$fit$objective+1e-4
 sims <- as.matrix(simulate(fit,nsim=500))
 zero_band <- quantile(colMeans(sims==0),c(.025,.975));var_band <- quantile(apply(sims,2,var),c(.025,.975))
 pi <- if(is_lab)plogis(fixef(fit)$zi[1]) else 0
 mu <- as.numeric(predict(fit,type="conditional"));y <- d$n_events
 pearson <- as.numeric(residuals(fit,type="pearson"));ord <- order(d$admission,d$period)
 same <- head(d$admission[ord],-1)==tail(d$admission[ord],-1)
 rows[[key]] <- data.frame(model=target$model,cc_group=target$cc_group,outcome=target$outcome,
  converged=converged,positive_hessian=isTRUE(fit$sdr$pdHess),eligible_retry=converged && no_worse,
  original_objective=old$fit$objective,retry_objective=fit$fit$objective,objective_change=fit$fit$objective-old$fit$objective,
  original_rate_ratio=exp(fixef(old)$cond["mhc1"]),rate_ratio=exp(beta),beta_log_rate=beta,standard_error=se,
  ci_low=exp(beta-1.96*se),ci_high=exp(beta+1.96*se),p_value=cf["mhc1",4],
  singular=any(vapply(vc,function(x)as.numeric(x[1,1])<1e-8,logical(1))),
  patient_variance=as.numeric(vc$patient[1,1]),pair_variance=as.numeric(vc$matched_pair[1,1]),admission_variance=as.numeric(vc$admission[1,1]),
  extra_zero_probability=as.numeric(pi),nb_theta=sigma(fit),AIC=AIC(fit),
  observed_zero_fraction=mean(y==0),conditional_expected_zero_fraction=mean(pi+(1-pi)*dnbinom(0,mu=mu,size=sigma(fit))),
  simulated_zero_low=zero_band[1],simulated_zero_high=zero_band[2],zero_screen_flag=mean(y==0)<zero_band[1] | mean(y==0)>zero_band[2],
  observed_variance=var(y),simulated_variance_low=var_band[1],simulated_variance_high=var_band[2],variance_screen_flag=var(y)<var_band[1] | var(y)>var_band[2],
  adjacent_pearson_correlation=cor(head(pearson[ord],-1)[same],tail(pearson[ord],-1)[same]),
  warnings=paste(unique(warnings),collapse=" | "))
 saveRDS(fit,file.path(out,paste0(gsub("[^a-zA-Z0-9_]","_",key),".rds")))
}
if(length(rows))write.csv(do.call(rbind,rows),file.path(out,"same_model_optimizer_comparison.csv"),row.names=FALSE)
writeLines(c("Same formulas and observations; only optimizer changes from nlminb to BFGS.",
 "Eligible only if optimizer succeeds, Hessian positive, and objective no worse than initial fit (tolerance 1e-4).",
 "Initial outputs remain intact; this review is not selection by significance or covariate specification."),file.path(out,"methods.txt"))
'''


def retry_adjusted_optimizers(output):
    r_library = PROJECT_DIR / "03_discharge_note_text_analysis" / "01_language_complexity" / ".r-library"
    subprocess.run(["Rscript", "-", str(output / "adjusted_interval_inputs.csv"), str(output), str(r_library)],
                   input=ADJUSTED_RETRY_R, text=True, check=True)


def fit_adjusted_longitudinal_models():
    """Join existing structured covariates and fit staged clinical models."""
    dataset = SCRIPT_DIR / "analysis_output_lab_linkage_and_timing" / "clinical_event_admission_intervals.csv"
    model_dir = PAIRED_WORKUP_OUTPUT_DIR / "longitudinal_negative_binomial_models"
    fingerprint = hashlib.sha256(dataset.read_bytes()).hexdigest()
    if (model_dir / "model_input_sha256.txt").read_text().strip() != fingerprint:
        raise ValueError("Current timing input differs from the baseline model input")
    covariate_path = PROJECT_DIR / "03_discharge_note_text_analysis" / "01_language_complexity" / "analysis_output_language_complexity_inference" / "race_ethnicity_sensitivity" / "model_covariates.csv"
    covariates = pd.read_csv(covariate_path)
    columns = ["age_at_admission_per_10y", "elixhauser_score_per_5pt", "log1p_prior_all_admissions", "language_group", "race_ethnicity_group"]
    keys = ["cohort", "subject_id", "hadm_id", "pair_id"]
    if covariates.duplicated(keys).any():
        raise ValueError("Duplicate admission covariates")
    original = pd.read_csv(dataset)
    original = original.loc[original.exposure_days.gt(0), keys + ["pure_cc_group", "exposure_days", "n_events", "interval", "measure"]]
    joined = original.merge(covariates[keys + columns], on=keys, how="left", validate="many_to_one", indicator=True)
    if not joined._merge.eq("both").all() or joined[columns].isna().any().any():
        raise ValueError("Missing adjusted model covariates; no silent admission exclusions allowed")
    output = model_dir / "adjusted_models"
    output.mkdir(parents=True, exist_ok=True)
    joined = joined.drop(columns="_merge")
    input_path = output / "adjusted_interval_inputs.csv"
    joined.to_csv(input_path, index=False)
    admissions = joined.drop_duplicates(keys)
    admissions.groupby(["pure_cc_group", "cohort", "language_group", "race_ethnicity_group"]).size().rename("n_admissions").reset_index().to_csv(output / "covariate_category_counts.csv", index=False)
    admissions.groupby(["pure_cc_group", "cohort"])[columns[:3]].agg(["count", "mean", "std", "min", "max"]).to_csv(output / "numeric_covariate_summary.csv")
    print(f"All covariates linked: {len(admissions):,} admissions, {admissions.pair_id.nunique():,} pairs; no covariate exclusions.", flush=True)
    input_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
    r_library = PROJECT_DIR / "03_discharge_note_text_analysis" / "01_language_complexity" / ".r-library"
    subprocess.run(["Rscript", "-", str(input_path), str(output), str(r_library)], input=ADJUSTED_LONGITUDINAL_R, text=True, check=True)
    if hashlib.sha256(input_path.read_bytes()).hexdigest() != input_hash:
        raise RuntimeError("Adjusted model inputs changed during fitting")
    diagnostics = pd.read_csv(output / "adjusted_longitudinal_diagnostics.csv")
    if len(diagnostics) != 80:
        raise RuntimeError("Expected diagnostics for all 80 adjusted models")
    if not diagnostics.converged.all():
        retry_adjusted_optimizers(output)
    finalize_adjusted_comparison(output, model_dir, fingerprint)
    (output / "model_input_sha256.txt").write_text(input_hash + "\n")
    (output / "source_timing_sha256.txt").write_text(fingerprint + "\n")
    (output / "source_covariates_sha256.txt").write_text(hashlib.sha256(covariate_path.read_bytes()).hexdigest() + "\n")


MARGINAL_LONGITUDINAL_R = r'''
args <- commandArgs(trailingOnly=TRUE)
.libPaths(c(args[3],.libPaths()))
first24 <- length(args)>=4 && args[4] %in% c("first24","first12")
window_hours <- if(first24)as.integer(sub("first","",args[4])) else NA_integer_
suppressPackageStartupMessages(library(sandwich))

# Inclusion-exclusion uses patient x pair intersections, NOT independent rows:
# several hospital-stay intervals occur in each intersection.
cluster_covariance <- function(fit,d) {
 v <- vcovCL(fit,cluster=d[,c("patient","matched_pair")],
             type="HC1",cadjust=TRUE,multi0=FALSE,fix=FALSE)
 x <- model.matrix(fit); mu <- fitted(fit); n <- nrow(x); k <- ncol(x)
 # Use the final IRLS weights used by glm's QR decomposition. They differ
 # slightly from fitted means at finite convergence tolerance.
 bread <- solve(crossprod(x,x*as.numeric(fit$weights)))
 scores <- x*as.numeric((fit$y-mu)/mu*fit$weights)
 component <- function(g) {
   s <- rowsum(scores,g,reorder=FALSE); G <- nrow(s)
   stopifnot(G>1)
   crossprod(s)*(G/(G-1))*((n-1)/(n-k))
 }
 meat <- component(d$patient)+component(d$matched_pair)-
         component(interaction(d$patient,d$matched_pair,drop=TRUE))
 manual <- bread %*% meat %*% bread
 stopifnot(isTRUE(all.equal(unname(v),unname(manual),tolerance=1e-7)))
 list(v=v,bread=bread,scores=scores,
      verification_error=max(abs(v-manual)))
}

# Small deterministic check independent of the clinical data.
set.seed(712)
test <- data.frame(patient=rep(as.vector(rbind(1:30,31+((1:30-1)%%15))),each=3),
                   matched_pair=rep(1:30,each=6),
                   mhc1=rep(rep(c(0,1),each=3),30),
                   period=rep(1:3,60),exposure_days=rep(c(1,.5,2),60))
test$n_events <- rpois(nrow(test),exp(.4+.2*test$mhc1-.1*test$period)*test$exposure_days)
test_fit <- glm(n_events~mhc1+factor(period)+offset(log(exposure_days)),
                data=test,family=poisson())
test_cov <- cluster_covariance(test_fit,test)
stopifnot(max(abs(test_cov$v-t(vcovCL(test_fit,
 cluster=test[,c("matched_pair","patient")],type="HC1",cadjust=TRUE,multi0=FALSE,fix=FALSE))))<1e-8)
cat("Synthetic two-way covariance checks passed.\n")
if(length(args)>=4 && args[4]=="self_test")quit(status=0)

data <- read.csv(args[1],check.names=FALSE); out <- args[2]
dir.create(out,recursive=TRUE,showWarnings=FALSE)
if(first24)data <- data[data$interval==paste0("0–",window_hours,"h"),]
stopifnot(!anyNA(data),all(data$exposure_days>0),
          all(is.finite(data$n_events)),all(data$n_events>=0),
          all(data$n_events==round(data$n_events)),
          all(data$cohort %in% c("MHC0","MHC1_psychotic")))
data$mhc1 <- as.integer(data$cohort=="MHC1_psychotic")
data$patient <- factor(data$subject_id); data$matched_pair <- factor(data$pair_id)
data$admission <- factor(data$hadm_id)
data$period <- factor(data$interval,levels=if(first24)paste0("0–",window_hours,"h") else c("0–24h","24–48h","48–72h","days 4–7","after day 7"))
data$language_group <- factor(data$language_group,levels=c("English","Non-English","Missing"))
data$race_ethnicity_group <- factor(data$race_ethnicity_group,
 levels=c("White","Black","Asian","Hispanic/Latino","Other recorded categories","Unknown/declined/missing"))
stopifnot(!anyNA(data),!anyDuplicated(data[,c("admission","measure","period")]))
if(first24)stopifnot(nrow(data)>0,all(data$exposure_days<=window_hours/24),!anyDuplicated(data[,c("admission","measure")]))
models <- list(M0_time_only=character(),
 M2_plus_prior_utilization=c("age_at_admission_per_10y","elixhauser_score_per_5pt","log1p_prior_all_admissions"),
 M3_plus_language=c("age_at_admission_per_10y","elixhauser_score_per_5pt","log1p_prior_all_admissions","language_group"),
 M4_plus_race_ethnicity=c("age_at_admission_per_10y","elixhauser_score_per_5pt","log1p_prior_all_admissions","language_group","race_ethnicity_group"))
effects <- list(); diagnostics <- list(); coefficients <- list(); calibration <- list(); influence <- list()
for(model in names(models)) for(group in sort(unique(data$pure_cc_group))) for(measure in sort(unique(data$measure))) {
 key <- paste(model,group,measure,sep="__")
 cat("Marginal PPML:",key,"\n"); flush.console()
 d <- droplevels(data[data$pure_cc_group==group & data$measure==measure,])
 admissions <- unique(d[,c("admission","patient","matched_pair","mhc1")])
 stopifnot(all(table(admissions$matched_pair)==2),
           all(tapply(admissions$mhc1,admissions$matched_pair,sum)==1))
 form <- as.formula(paste("n_events ~",paste(c("mhc1",if(!first24)"period",models[[model]],
                                               "offset(log(exposure_days))"),collapse=" + ")))
 warnings <- character()
 fit <- withCallingHandlers(glm(form,data=d,family=poisson(link="log"),
                   na.action=na.fail,control=glm.control(maxit=100,epsilon=1e-10)),
   warning=function(w){warnings <<- c(warnings,conditionMessage(w));invokeRestart("muffleWarning")})
 x <- model.matrix(fit)
 stopifnot(nobs(fit)==nrow(d),fit$rank==ncol(x),all(is.finite(coef(fit))))
 cov <- cluster_covariance(fit,d); v <- cov$v
 eigenvalues <- eigen(v,symmetric=TRUE,only.values=TRUE)$values
 psd <- min(eigenvalues)>= -1e-8*max(abs(eigenvalues))
 valid <- isTRUE(fit$converged) && all(is.finite(v)) && all(diag(v)>0) && psd
 se <- sqrt(pmax(diag(v),0)); beta <- coef(fit)
 z <- beta/se; p <- 2*pnorm(-abs(z))
 low <- beta-qnorm(.975)*se; high <- beta+qnorm(.975)*se
 if(!valid){p[] <- NA_real_; low[] <- NA_real_; high[] <- NA_real_}
 coefficients[[key]] <- data.frame(model=model,cc_group=group,outcome=measure,
   term=names(beta),beta=as.numeric(beta),cluster_robust_se=se,
   ci_low=low,ci_high=high,rate_ratio=exp(beta),rr_ci_low=exp(low),rr_ci_high=exp(high),
   p_value=p,inference_valid=valid,row.names=NULL)
effects[[key]] <- data.frame(model=model,cc_group=group,outcome=measure,
   n_intervals=nrow(d),n_admissions=nlevels(d$admission),n_pairs=nlevels(d$matched_pair),
   n_patients=nlevels(d$patient),beta_log_rate=beta["mhc1"],standard_error=se["mhc1"],
   rate_ratio=exp(beta["mhc1"]),ci_low=exp(low["mhc1"]),ci_high=exp(high["mhc1"]),
   p_value=p["mhc1"],converged=fit$converged,inference_valid=valid,row.names=NULL)
 mu <- fitted(fit); residual <- d$n_events-mu
 # Point-estimate influence audit: rank cluster scores, then refit after removing
 # three highest-ranked clusters. Patient checks remove their entire matched
 # pairs, so this diagnostic never leaves unmatched controls/cases behind.
 # Primary estimates retain every observation. No diagnostic deletion p-values.
 exact_changes <- numeric()
 for(kind in c("patient","matched_pair")) {
   cluster_scores <- rowsum(cov$scores,d[[kind]],reorder=FALSE)
   approx_change <- -as.numeric(cluster_scores %*% cov$bread[,"mhc1"])
   top <- head(order(abs(approx_change),decreasing=TRUE),3)
   for(rank in seq_along(top)) {
     cluster <- rownames(cluster_scores)[top[rank]]
     pairs_to_drop <- unique(d$matched_pair[as.character(d[[kind]])==cluster])
     keep <- !d$matched_pair %in% pairs_to_drop
     reduced <- tryCatch(suppressWarnings(glm(form,data=d[keep,],family=poisson(),
                      control=glm.control(maxit=100,epsilon=1e-10))),error=function(e)NULL)
     ok <- !is.null(reduced) && isTRUE(reduced$converged) && is.finite(coef(reduced)["mhc1"])
     b <- if(ok)coef(reduced)["mhc1"] else NA_real_
     change <- b-beta["mhc1"]; exact_changes <- c(exact_changes,change)
     influence[[paste(key,kind,rank)]] <- data.frame(model=model,cc_group=group,outcome=measure,
       ranked_cluster_type=kind,rank=rank,diagnostic_pairs_removed=length(pairs_to_drop),
       diagnostic_admissions_removed=length(unique(d$admission[!keep])),
       primary_rate_ratio=exp(beta["mhc1"]),diagnostic_rate_ratio=exp(b),
       beta_change=change,change_in_primary_se=change/se["mhc1"],refit_converged=ok,row.names=NULL)
   }
 }
 by_patient <- rowsum(cbind(events=d$n_events,exposure=d$exposure_days),d$patient,reorder=FALSE)
 top_patient <- head(order(by_patient[,"events"],decreasing=TRUE),max(1,ceiling(nrow(by_patient)*.01)))
 diagnostics[[key]] <- data.frame(model=model,cc_group=group,outcome=measure,
   converged=fit$converged,inference_valid=valid,covariance_psd=psd,
   min_covariance_eigenvalue=min(eigenvalues),covariance_verification_error=cov$verification_error,
   n_patients=nlevels(d$patient),n_pairs=nlevels(d$matched_pair),
   n_patient_pair_intersections=nlevels(interaction(d$patient,d$matched_pair,drop=TRUE)),
   max_interval_leverage=max(hatvalues(fit)),
   top_one_percent_patients_event_share=sum(by_patient[top_patient,"events"])/sum(d$n_events),
   top_one_percent_patients_exposure_share=sum(by_patient[top_patient,"exposure"])/sum(d$exposure_days),
   max_selected_deletion_beta_change=max(abs(exact_changes),na.rm=TRUE),
   max_selected_deletion_change_in_se=max(abs(exact_changes),na.rm=TRUE)/se["mhc1"],
   warnings=paste(unique(warnings),collapse=" | "),row.names=NULL)
 for(cohort in levels(factor(d$cohort))) for(period in levels(d$period)) {
   ix <- which(d$cohort==cohort & d$period==period); if(!length(ix))next
   days <- sum(d$exposure_days[ix])
   calibration[[paste(key,cohort,period)]] <- data.frame(model=model,cc_group=group,outcome=measure,
     cohort=cohort,interval=period,n_intervals=length(ix),observed_days=days,
     observed_events=sum(d$n_events[ix]),predicted_events=sum(mu[ix]),
     observed_events_per_day=sum(d$n_events[ix])/days,predicted_events_per_day=sum(mu[ix])/days,
     observed_to_predicted=sum(d$n_events[ix])/sum(mu[ix]))
 }
 if(first24) {
   # Cohort-total calibration is exact by the GLM score equations. Examine
   # predicted-count bins and short/full windows instead; descriptive only.
   breaks <- unique(as.numeric(quantile(mu,seq(0,1,.2))))
   fitted_bin <- if(length(breaks)>1)as.character(cut(mu,breaks=breaks,include.lowest=TRUE)) else rep("constant fitted count",length(mu))
   for(kind in c("fitted_count_bin","window_duration")) {
     band <- if(kind=="fitted_count_bin")fitted_bin else ifelse(d$exposure_days<window_hours/24,paste0("less than ",window_hours,"h"),paste0("full ",window_hours,"h"))
     for(cohort in unique(d$cohort)) for(label in unique(band)) {
       ix <- which(d$cohort==cohort & band==label); if(!length(ix))next
       calibration[[paste(key,kind,cohort,label)]] <- data.frame(model=model,cc_group=group,outcome=measure,
         cohort=cohort,interval=paste(kind,label,sep=": "),n_intervals=length(ix),observed_days=sum(d$exposure_days[ix]),
         observed_events=sum(d$n_events[ix]),predicted_events=sum(mu[ix]),
         observed_events_per_day=sum(d$n_events[ix])/sum(d$exposure_days[ix]),
         predicted_events_per_day=sum(mu[ix])/sum(d$exposure_days[ix]),
         observed_to_predicted=sum(d$n_events[ix])/sum(mu[ix]))
     }
   }
 }
 saveRDS(list(fit=fit,cluster_covariance=v),file.path(out,paste0(gsub("[^a-zA-Z0-9_]","_",key),".rds")))
}
r <- do.call(rbind,effects); r$FDR <- NA_real_
for(model in names(models)) {
 ix <- which(r$model==model & r$inference_valid & is.finite(r$p_value))
 r$FDR[ix] <- p.adjust(r$p_value[ix],method="BH",n=20)
}
write.csv(r,file.path(out,"marginal_cohort_effects.csv"),row.names=FALSE)
write.csv(do.call(rbind,coefficients),file.path(out,"marginal_coefficients.csv"),row.names=FALSE)
write.csv(do.call(rbind,diagnostics),file.path(out,"marginal_diagnostics.csv"),row.names=FALSE)
write.csv(do.call(rbind,calibration),file.path(out,"mean_calibration_by_cohort_period.csv"),row.names=FALSE)
write.csv(do.call(rbind,influence),file.path(out,"selected_cluster_influence_checks.csv"),row.names=FALSE)
writeLines(c(
 "Population-average log-mean PPML models, separately for five CCs and four recorded-event outcomes.",
 "M0: cohort + categorical hospital period + log(observed days) offset.",
 "M2 adds age/10, Elixhauser/5, log1p(prior admissions); M3 adds language; M4 adds race/ethnicity to M3.",
 "Same saved positive-exposure intervals and complete timing pairs for every stage; no exclusions or new imputation.",
 "Patient and matched-pair two-way sandwich covariance: HC1, G/(G-1) cluster corrections, intersection subtraction.",
 "Intersections contain repeated intervals: multi0=FALSE. No eigenvalue clipping (fix=FALSE).",
 "Independent direct score/Hessian calculation verified against sandwich::vcovCL for every fit and synthetic data.",
 "Asymptotic normal Wald intervals/p-values; BH across 20 cohort effects separately for M0, M2, M3, M4.",
 "No random intercepts, no zero-inflation component, no Poisson equidispersion assumption for robust inference.",
 "Assumes correctly specified expected count, adequate independent clusters, finite relevant moments and one common cohort rate ratio across periods.",
 "Patient clustering accommodates within-admission intervals and readmissions; pair clustering handles shared-pair dependence.",
 "Clustering changes uncertainty, not the mean estimate. This is not a pair-fixed-effect or patient-specific conditional comparison.",
 "No AIC/distribution-zero pass/fail selection. Mean calibration and leverage/influence are descriptive checks, not validation certificates.",
 "Influence checks rank patient/pair aggregated scores, refit dropping the top three clusters separately; patient deletions drop all affected pairs.",
 "Only diagnostic refits omit observations. Primary models retain extremes. Selected checks are not exhaustive leave-one-cluster-out analysis.",
 "Later intervals concern patients still hospitalized. Informative discharge, time-varying severity and residual confounding remain unaddressed.",
 "Counts are recorded laboratory measurements, specimens, microbiology records and provider-order transactions, not care quality or unique clinical decisions.",
 "English and White references. Missing/unknown language/race categories preserved. Existing mixed-model outputs unchanged.",
 "References: https://www.jstatsoft.org/article/view/v095i01 ; https://personal.lse.ac.uk/tenreyro/LGW.html"
),file.path(out,"marginal_methods.txt"))
if(first24)writeLines(c(
 "First-24-hour sensitivity: one observation per admission and outcome, window [admittime,min(admittime+24h,dischtime)).",
 "Uses existing timestamped-event counts; no calendar-day or midnight imputation. Same complete timing pairs as full-stay models.",
 "M0: cohort + log(observed days) offset; no hospital-period term because only one period remains.",
 "M2 adds age/10, Elixhauser/5 and log1p(prior admissions); M3 adds language; M4 adds race/ethnicity to M3.",
 "PPML with patient/pair two-way HC1 sandwich covariance, cluster corrections and intersection subtraction; independently checked for every fit.",
 "No random intercepts or zero-inflation component. Asymptotic normal Wald intervals; BH across 20 cohort comparisons separately per stage.",
 "Stays shorter than 24h retained with actual-duration offsets. This estimates rates during observed early inpatient time, not identical full-24h counts for everyone.",
 "Offset assumes proportional expected counts to observed duration within this window conditional on covariates; front-loaded activity/early discharge can violate it.",
 "Missing exact microbiology timestamps are excluded from window counts. Source timestamp audit reports date-only rows; chartdate cannot place events precisely in this rolling window.",
 "Zero count means no included timestamped events in the window, not proof of no care or no date-only events.",
 "Covariance validity/convergence do not certify mean fit. Calibration by cohort alone is mechanically exact; predicted-count quintiles and short/full-duration bands are descriptive checks.",
 "Same score-ranked patient/pair influence checks as full-stay models; only diagnostic refits omit observations, preserving affected pairs. Primary fits retain all extremes.",
 "Repeated admissions still require patient clustering even with one row per admission. Pair clustering accounts for shared-pair dependence without pair fixed effects.",
 "Counts are recorded laboratory measurements/specimens, microbiology rows and provider-order transactions. Early inpatient time excludes pre-admission ED events.",
 "First-24h analysis is exploratory; original full-stay results remain available. Language/race are sensitivity extensions.",
 "References: https://www.jstatsoft.org/article/view/v095i01 ; https://mimic.mit.edu/docs/iv/modules/hosp/microbiologyevents.html"
),file.path(out,"marginal_methods.txt"))
if(first24 && window_hours==12) {
 methods <- readLines(file.path(out,"marginal_methods.txt"))
 writeLines(gsub("24","12",methods,fixed=TRUE),file.path(out,"marginal_methods.txt"))
}
capture.output(sessionInfo(),file=file.path(out,"R_session_info.txt"))
print(aggregate(cbind(converged,inference_valid)~model,data=r,FUN=sum))
'''


def build_first12_inputs(source_path: Path, output: Path) -> Path:
    """Recount timestamped events; reproduce saved 24h counts before fitting 12h."""
    import duckdb

    base = pd.read_csv(source_path)
    base = base[base.interval.eq("0–24h")].copy()
    keys = ["cohort", "subject_id", "hadm_id"]
    selected = base[keys].drop_duplicates().merge(
        load_descriptors()[keys + ["admittime", "dischtime"]], on=keys,
        validate="one_to_one", how="left")
    if selected[["admittime", "dischtime"]].isna().any().any():
        raise ValueError("Missing admission timestamps for first-12h recount")
    if not selected.dischtime.gt(selected.admittime).all():
        raise ValueError("First-12h recount requires positive inpatient duration")
    with duckdb.connect(str(common.DB_PATH), read_only=True) as con:
        con.register("early_admissions", selected)
        # Only admission/event identifiers and timestamps enter these aggregates.
        # NULL-hadm labs use the same globally unique inpatient linkage as 06.
        counts = con.execute("""
        WITH linked AS MATERIALIZED (
            SELECT s.subject_id,s.hadm_id,l.specimen_id,l.charttime
            FROM early_admissions s JOIN labevents l
            ON s.subject_id=l.subject_id AND s.hadm_id=l.hadm_id
        ), recovered AS MATERIALIZED (
            SELECT s.subject_id,s.hadm_id,l.specimen_id,l.charttime
            FROM early_admissions s JOIN labevents l ON s.subject_id=l.subject_id
            WHERE l.hadm_id IS NULL AND l.charttime>=s.admittime AND l.charttime<s.dischtime
            AND NOT EXISTS (SELECT 1 FROM admissions a
                WHERE a.subject_id=l.subject_id AND a.hadm_id<>s.hadm_id
                AND a.dischtime>a.admittime AND l.charttime>=a.admittime AND l.charttime<a.dischtime)
        ), labs AS MATERIALIZED (SELECT * FROM linked UNION ALL SELECT * FROM recovered),
        lab_counts AS (
            SELECT s.subject_id,s.hadm_id,
                count(*) FILTER (WHERE l.charttime<s.admittime+INTERVAL '12 hours') AS lab12,
                count(*) FILTER (WHERE l.charttime<s.admittime+INTERVAL '24 hours') AS lab24
            FROM early_admissions s JOIN labs l USING(subject_id,hadm_id)
            WHERE l.charttime>=s.admittime AND l.charttime<s.dischtime GROUP BY s.subject_id,s.hadm_id
        ), specimens AS (
            SELECT subject_id,hadm_id,specimen_id,min(charttime) AS charttime FROM labs
            WHERE specimen_id IS NOT NULL GROUP BY subject_id,hadm_id,specimen_id
        ), specimen_counts AS (
            SELECT s.subject_id,s.hadm_id,
                count(*) FILTER (WHERE l.charttime<s.admittime+INTERVAL '12 hours') AS specimen12,
                count(*) FILTER (WHERE l.charttime<s.admittime+INTERVAL '24 hours') AS specimen24
            FROM early_admissions s JOIN specimens l USING(subject_id,hadm_id)
            WHERE l.charttime>=s.admittime AND l.charttime<s.dischtime GROUP BY s.subject_id,s.hadm_id
        ), micro_counts AS (
            SELECT s.subject_id,s.hadm_id,
                count(*) FILTER (WHERE m.charttime<s.admittime+INTERVAL '12 hours') AS micro12,
                count(*) FILTER (WHERE m.charttime<s.admittime+INTERVAL '24 hours') AS micro24
            FROM early_admissions s JOIN microbiologyevents m USING(subject_id,hadm_id)
            WHERE m.charttime>=s.admittime AND m.charttime<s.dischtime GROUP BY s.subject_id,s.hadm_id
        ), poe_counts AS (
            SELECT s.subject_id,s.hadm_id,
                count(*) FILTER (WHERE p.ordertime<s.admittime+INTERVAL '12 hours') AS poe12,
                count(*) FILTER (WHERE p.ordertime<s.admittime+INTERVAL '24 hours') AS poe24
            FROM early_admissions s JOIN poe p USING(subject_id,hadm_id)
            WHERE p.ordertime>=s.admittime AND p.ordertime<s.dischtime GROUP BY s.subject_id,s.hadm_id
        ) SELECT s.cohort,s.subject_id,s.hadm_id,
            coalesce(l.lab12,0) AS lab12,coalesce(l.lab24,0) AS lab24,
            coalesce(sp.specimen12,0) AS specimen12,coalesce(sp.specimen24,0) AS specimen24,
            coalesce(m.micro12,0) AS micro12,coalesce(m.micro24,0) AS micro24,
            coalesce(p.poe12,0) AS poe12,coalesce(p.poe24,0) AS poe24
        FROM early_admissions s LEFT JOIN lab_counts l USING(subject_id,hadm_id)
        LEFT JOIN specimen_counts sp USING(subject_id,hadm_id)
        LEFT JOIN micro_counts m USING(subject_id,hadm_id)
        LEFT JOIN poe_counts p USING(subject_id,hadm_id)
        """).fetchdf()
    pieces = []
    qc = []
    for measure, stem in [("laboratory_measurements", "lab"), ("laboratory_specimens", "specimen"),
                          ("microbiology_records", "micro"), ("provider_order_activity", "poe")]:
        f = base[base.measure.eq(measure)].merge(counts[keys + [stem + "12", stem + "24"]],
                on=keys, how="left", validate="one_to_one")
        if f[stem + "12"].isna().any() or not f.n_events.eq(f[stem + "24"]).all():
            raise ValueError(f"Independent 24h reconstruction differs from saved counts: {measure}")
        if not f[stem + "12"].le(f[stem + "24"]).all():
            raise ValueError("First-12h counts exceed first-24h counts")
        qc.append(dict(outcome=measure,n_admissions=len(f),n_24h_count_mismatches=0,
                       total_events_first12=int(f[stem + "12"].sum()),total_events_first24=int(f[stem + "24"].sum())))
        f["n_events"] = f[stem + "12"].astype(int)
        f["exposure_days"] = f.exposure_days.clip(upper=.5)
        f["interval"] = "0–12h"
        pieces.append(f[base.columns])
    result = pd.concat(pieces, ignore_index=True)
    path = output / "first12_interval_inputs.csv"
    result.to_csv(path, index=False)
    pd.DataFrame(qc).to_csv(output / "independent_event_recount_qc.csv", index=False)
    print("Independent 24h reconstruction agrees for every admission/outcome; 12h counts prepared.", flush=True)
    return path


def fit_marginal_longitudinal_models(first24: bool = False, first12: bool = False):
    """PPML with patient/pair sandwich inference; preserve all mixed-model outputs."""
    import numpy as np
    from statsmodels.stats.multitest import multipletests

    model_dir = PAIRED_WORKUP_OUTPUT_DIR / "longitudinal_negative_binomial_models"
    adjusted = model_dir / "adjusted_models"
    input_path = adjusted / "adjusted_interval_inputs.csv"
    fingerprint = hashlib.sha256(input_path.read_bytes()).hexdigest()
    if (adjusted / "model_input_sha256.txt").read_text().strip() != fingerprint:
        raise ValueError("Saved mixed models use different interval inputs")
    source_fingerprint = fingerprint
    hours = 12 if first12 else 24
    first24 = first24 or first12
    window_stem = f"first{hours}"
    output = PAIRED_WORKUP_OUTPUT_DIR / (f"{window_stem}_marginal_models" if first24 else "longitudinal_marginal_models")
    output.mkdir(parents=True, exist_ok=True)
    if first12:
        input_path = build_first12_inputs(input_path, output)
        fingerprint = hashlib.sha256(input_path.read_bytes()).hexdigest()
    if first24:
        windows = pd.read_csv(input_path)
        windows = windows[windows.interval.eq(f"0–{hours}h")].copy()
        if windows.duplicated(["cohort", "hadm_id", "measure"]).any():
            raise ValueError("First-24h input must have one observation per admission/outcome")
        descriptors = windows.drop_duplicates(["cohort", "subject_id", "hadm_id", "pair_id"])
        if not descriptors.groupby("pair_id").size().eq(2).all():
            raise ValueError("First-24h input lost complete matching pairs")
        descriptors.groupby(["pure_cc_group", "cohort"]).agg(
            n_admissions=("hadm_id", "nunique"), n_patients=("subject_id", "nunique"),
            n_shorter_than_window=("exposure_days", lambda x: x.lt(hours/24).sum()),
            mean_observed_hours=("exposure_days", lambda x: x.mean()*24),
            minimum_observed_hours=("exposure_days", lambda x: x.min()*24)
        ).reset_index().to_csv(output / "window_selection_qc.csv", index=False)
        windows.groupby(["pure_cc_group", "cohort", "measure"]).agg(
            n_admissions=("hadm_id", "nunique"), total_events=("n_events", "sum"),
            mean_count=("n_events", "mean"), median_count=("n_events", "median"),
            q1_count=("n_events", lambda x: x.quantile(.25)), q3_count=("n_events", lambda x: x.quantile(.75)),
            n_zero_counts=("n_events", lambda x: x.eq(0).sum()), observed_days=("exposure_days", "sum")
        ).reset_index().assign(pooled_events_per_day=lambda x: x.total_events/x.observed_days).to_csv(
            output / f"{window_stem}_event_descriptives.csv", index=False)
        import duckdb
        with duckdb.connect(str(common.DB_PATH), read_only=True) as con:
            con.register("first24_admissions", descriptors)
            audit = con.execute("""SELECT a.pure_cc_group,a.cohort,count(*) AS linked_microbiology_rows,
                count(*) FILTER (WHERE m.charttime IS NULL) AS missing_exact_timestamp_rows,
                count(*) FILTER (WHERE m.charttime IS NULL AND m.chartdate IS NOT NULL) AS date_only_rows
                FROM first24_admissions a JOIN microbiologyevents m
                ON a.subject_id=m.subject_id AND a.hadm_id=m.hadm_id
                GROUP BY a.pure_cc_group,a.cohort""").fetchdf()
        audit.to_csv(output / "microbiology_timestamp_source_audit.csv", index=False)
        saved_audit = SCRIPT_DIR / "analysis_output_lab_linkage_and_timing" / "clinical_event_timestamp_audit.csv"
        if saved_audit.exists():
            scope = pd.read_csv(saved_audit)
            scope.to_csv(output / "source_event_timestamp_scope_audit.csv", index=False)
    r_library = PROJECT_DIR / "03_discharge_note_text_analysis" / "01_language_complexity" / ".r-library"
    command = ["Rscript", "-", str(input_path), str(output), str(r_library)]
    if first24:
        command.append(window_stem)
    subprocess.run(command,
                   input=MARGINAL_LONGITUDINAL_R, text=True, check=True)
    if hashlib.sha256(input_path.read_bytes()).hexdigest() != fingerprint:
        raise RuntimeError("Inputs changed while fitting marginal models")
    effects = pd.read_csv(output / "marginal_cohort_effects.csv")
    diagnostics = pd.read_csv(output / "marginal_diagnostics.csv")
    if len(effects) != 80 or len(diagnostics) != 80:
        raise RuntimeError("Expected 80 marginal fits and diagnostic rows")
    for _, family in effects.groupby("model"):
        assert len(family) == 20
        usable = family.inference_valid & family.p_value.notna()
        planned_p = family.loc[usable, "p_value"].tolist() + [1.] * (20 - usable.sum())
        expected = multipletests(planned_p, method="fdr_bh")[1][:usable.sum()]
        np.testing.assert_allclose(family.loc[usable, "FDR"], expected, rtol=1e-10, atol=1e-12)
    baseline = pd.read_csv(model_dir / "zero_inflated_laboratory_models" / "exploratory_20_outcome_family.csv")
    baseline["model"] = "M0_time_only"
    old_adjusted = pd.read_csv(adjusted / "optimizer_checked_cohort_effects.csv")
    old_adjusted = old_adjusted[old_adjusted.model.isin(effects.model.unique())]
    original = pd.concat([baseline, old_adjusted], ignore_index=True)
    keys = ["model", "cc_group", "outcome"]
    fields = ["n_intervals", "n_admissions", "n_pairs", "n_patients", "rate_ratio", "ci_low", "ci_high", "FDR", "converged"]
    comparison = original[keys + fields].merge(effects[keys + fields], on=keys,
        suffixes=("_mixed", "_marginal"), validate="one_to_one")
    assert len(comparison) == 80
    for field in fields[1:4] if first24 else fields[:4]:
        if not comparison[field + "_mixed"].eq(comparison[field + "_marginal"]).all():
            raise ValueError("Marginal and mixed models do not use identical sample sizes")
    comparison.loc[~comparison.converged_mixed, ["ci_low_mixed", "ci_high_mixed", "FDR_mixed"]] = np.nan
    comparison.to_csv(output / ("full_stay_mixed_vs_first24_marginal.csv" if first24 else "mixed_vs_marginal_comparison.csv"), index=False)
    recent_path = model_dir / "without_admission_intercept" / "M0_M2_reporting.csv"
    if recent_path.exists() and not first24:
        recent_hash = (recent_path.parent / "model_input_sha256.txt").read_text().strip()
        if recent_hash != fingerprint:
            raise ValueError("No-admission-intercept sensitivity uses different inputs")
        recent = pd.read_csv(recent_path)
        recent_comparison = recent[keys + fields].merge(effects[keys + fields], on=keys,
            suffixes=("_mixed_without_admission", "_marginal"), validate="one_to_one")
        assert len(recent_comparison) == 40
        for field in fields[:4]:
            assert recent_comparison[field + "_mixed_without_admission"].eq(recent_comparison[field + "_marginal"]).all()
        recent_comparison.to_csv(output / "no_admission_mixed_vs_marginal_comparison.csv", index=False)
    effects[effects.model.isin(["M0_time_only", "M2_plus_prior_utilization"])].to_csv(output / "M0_M2_reporting.csv", index=False)
    (output / "model_input_sha256.txt").write_text(fingerprint + "\n")
    (output / "analysis_source_sha256.txt").write_text(hashlib.sha256(MARGINAL_LONGITUDINAL_R.encode()).hexdigest() + "\n")
    (output / "analysis_window.txt").write_text(window_stem + "\n" if first24 else "full_stay\n")
    (output / "source_full_interval_sha256.txt").write_text(source_fingerprint + "\n")
    if first24:
        full_path = PAIRED_WORKUP_OUTPUT_DIR / "longitudinal_marginal_models" / "marginal_cohort_effects.csv"
        if (full_path.parent / "model_input_sha256.txt").read_text().strip() != source_fingerprint:
            raise ValueError("Full-stay marginal models use different source inputs")
        full = pd.read_csv(full_path)
        comparison = full[keys + fields].merge(effects[keys + fields], on=keys,
            suffixes=("_full_stay", "_first24"), validate="one_to_one")
        assert len(comparison) == 80
        for field in fields[1:4]:
            assert comparison[field + "_full_stay"].eq(comparison[field + "_first24"]).all()
        comparison.to_csv(output / "full_stay_vs_first24_marginal.csv", index=False)
    summary = ["# Marginal clinical-activity models", "",
        "Same saved count/covariate input as the mixed models. No primary exclusions or rematching.",
        "Rate ratios compare MHC1-psychosis with MHC0. Values below 1 indicate lower recorded activity.",
        "95% confidence intervals are pointwise; FDR is BH-adjusted across 20 cohort comparisons per model stage.",
        "M0 includes cohort and period; M2 additionally age, Elixhauser and prior admissions.",
        "M3 adds language to M2; M4 adds race/ethnicity to M3. All use observed-time offsets and patient/pair clustered uncertainty.", "",
        "| Model | Converged | Covariance checks passed | FDR < 0.05 |", "|---|---:|---:|---:|"]
    for model, family in effects.groupby("model", sort=False):
        summary.append(f"| {model} | {family.converged.sum()}/20 | {family.inference_valid.sum()}/20 | {family.FDR.lt(.05).sum()} |")
    for model in ["M0_time_only", "M2_plus_prior_utilization"]:
        summary += ["", f"## {model}", "", "| CC | Outcome | Rate ratio [95% CI] | FDR |", "|---|---|---:|---:|"]
        for row in effects[effects.model.eq(model)].itertuples():
            fdr = "n.a." if pd.isna(row.FDR) else ("<0.001" if row.FDR < .001 else f"{row.FDR:.3f}")
            estimate = f"{row.rate_ratio:.3f} [{row.ci_low:.3f}, {row.ci_high:.3f}]" if row.inference_valid else "Inference unavailable"
            summary.append(f"| {row.cc_group} | {row.outcome.replace('_', ' ')} | {estimate} | {fdr} |")
    summary += ["", "## Interpretation and checks", "",
        "Numerical convergence and a valid covariance matrix do not establish correct mean specification.",
        "Inspect mean_calibration_by_cohort_period.csv for observed versus predicted rates. No automatic zero-frequency/variance pass-fail screen is used for PPML.",
        "selected_cluster_influence_checks.csv contains diagnostic refits only. All primary fits retain every admission, including extremes.",
        "These checks inspect the three largest score-influence candidates per patient/pair dimension, not every possible deletion.",
        "Mixed-model comparisons concern different modelling assumptions/estimands and are not model-selection tests.",
        "Later periods describe patients still hospitalized. Recorded event rates do not establish equal clinical need, quality of care or diagnostic overshadowing.", ""]
    (output / "results_summary.md").write_text("\n".join(summary))
    if first24:
        summary[0] = f"# First-{hours}-hour clinical-activity models"
        summary[5] = "M0 includes cohort only; M2 additionally age, Elixhauser and prior admissions. There is no hospital-period term."
        # Replace full-stay interpretation with the rolling-window definition.
        summary = [line.replace("Later periods describe patients still hospitalized.",
            "Short stays contribute their actual duration before discharge; the offset assumes proportional activity to observed duration within the window.") for line in summary]
        summary[2] = f"One observation per admission/outcome during [admission, min(admission + {hours}h, discharge)). Same matched sample as the full-stay models."
        summary += ["", f"{len(descriptors):,} admissions / {descriptors.pair_id.nunique():,} pairs; {descriptors.exposure_days.lt(hours/24).sum():,} admissions contribute less than {hours}h.",
            f"Source audit: {audit.missing_exact_timestamp_rows.sum():,} linked microbiology rows lack an exact timestamp; {audit.date_only_rows.sum():,} have a date only.",
            "Cohort-total calibration is mechanically exact; use the quintile and duration-band rows for descriptive mean-fit checks.", ""]
        (output / "results_summary.md").write_text("\n".join(summary))
    print("Marginal results saved separately:", output)
    print(effects.groupby("model")[["converged", "inference_valid"]].sum().to_string())


def fit_without_admission_intercept():
    """Apply a uniform patient/pair-only simplification to M0 and M2."""
    model_dir = PAIRED_WORKUP_OUTPUT_DIR / "longitudinal_negative_binomial_models"
    adjusted = model_dir / "adjusted_models"
    input_path = adjusted / "adjusted_interval_inputs.csv"
    fingerprint = hashlib.sha256(input_path.read_bytes()).hexdigest()
    if (adjusted / "model_input_sha256.txt").read_text().strip() != fingerprint:
        raise ValueError("Saved full-intercept models use different interval inputs")
    output = model_dir / "without_admission_intercept"
    r_library = PROJECT_DIR / "03_discharge_note_text_analysis" / "01_language_complexity" / ".r-library"
    subprocess.run(["Rscript", "-", str(input_path), str(output), str(r_library), "no_admission_M0_M2"],
                   input=ADJUSTED_LONGITUDINAL_R, text=True, check=True)
    if hashlib.sha256(input_path.read_bytes()).hexdigest() != fingerprint:
        raise RuntimeError("Inputs changed during random-effects sensitivity")
    effects = pd.read_csv(output / "adjusted_longitudinal_cohort_effects.csv")
    diag = pd.read_csv(output / "adjusted_longitudinal_diagnostics.csv")
    if len(effects) != 40 or len(diag) != 40:
        raise RuntimeError("Expected forty M0/M2 fits and diagnostics")
    baseline = pd.read_csv(model_dir / "zero_inflated_laboratory_models" / "exploratory_20_outcome_family.csv")
    baseline["model"] = "M0_time_only"
    m2 = pd.read_csv(adjusted / "optimizer_checked_cohort_effects.csv")
    original = pd.concat([baseline, m2[m2.model.eq("M2_plus_prior_utilization")]], ignore_index=True)
    keys = ["model", "cc_group", "outcome"]
    values = ["n_intervals", "n_admissions", "n_pairs", "n_patients", "rate_ratio", "ci_low", "ci_high", "FDR", "converged", "singular"]
    comparison = original[keys+values].merge(effects[keys+values], on=keys,
        suffixes=("_full", "_no_admission"), validate="one_to_one")
    for field in ["n_intervals", "n_admissions", "n_pairs", "n_patients"]:
        if not comparison[field+"_full"].eq(comparison[field+"_no_admission"]).all():
            raise ValueError("Random-effects comparison has different observations")
    old_nb = pd.read_csv(model_dir / "longitudinal_diagnostics.csv")
    old_zi = pd.read_csv(model_dir / "zero_inflated_laboratory_models" / "zero_inflated_diagnostics.csv")
    old_nb = old_nb[~old_nb.outcome.isin(["laboratory_measurements", "laboratory_specimens"])].copy()
    old_nb = old_nb.rename(columns={"adjacent_conditional_pearson_correlation": "adjacent_pearson_correlation"})
    old_zi = old_zi.rename(columns={"ZI_AIC": "AIC"})
    old_m0 = pd.concat([old_nb,old_zi],ignore_index=True)
    old_m0["model"] = "M0_time_only"
    old_adjusted = pd.read_csv(adjusted / "optimizer_checked_diagnostics.csv")
    old_diag = pd.concat([old_m0,old_adjusted[old_adjusted.model.eq("M2_plus_prior_utilization")]],ignore_index=True)
    fields = ["AIC", "zero_screen_flag", "variance_screen_flag", "adjacent_pearson_correlation"]
    dc = old_diag[keys+fields].merge(diag[keys+fields],on=keys,suffixes=("_full", "_no_admission"),validate="one_to_one")
    comparison = comparison.merge(dc,on=keys,validate="one_to_one")
    comparison["AIC_change_no_admission_minus_full"] = comparison.AIC_no_admission - comparison.AIC_full
    comparison.loc[~(comparison.converged_full & comparison.converged_no_admission), "AIC_change_no_admission_minus_full"] = float("nan")
    comparison.to_csv(output / "full_vs_no_admission_comparison.csv",index=False)
    diag.groupby("model")[["converged", "singular", "zero_screen_flag", "variance_screen_flag"]].sum().to_csv(output / "diagnostic_flag_summary.csv")
    report = effects.copy()
    report.loc[~report.converged,["ci_low", "ci_high", "p_value", "FDR"]] = float("nan")
    report.to_csv(output / "M0_M2_reporting.csv",index=False)
    (output / "model_input_sha256.txt").write_text(fingerprint+"\n")
    print(diag.groupby("model")[["converged", "singular", "zero_screen_flag", "variance_screen_flag"]].sum().to_string())


def fit_m2_distribution_comparison():
    """Fit ordinary M2 labs and compare against verified saved ZI M2 fits."""
    import numpy as np

    model_dir = PAIRED_WORKUP_OUTPUT_DIR / "longitudinal_negative_binomial_models"
    adjusted = model_dir / "adjusted_models"
    input_path = adjusted / "adjusted_interval_inputs.csv"
    fingerprint = hashlib.sha256(input_path.read_bytes()).hexdigest()
    if (adjusted / "model_input_sha256.txt").read_text().strip() != fingerprint:
        raise ValueError("Saved adjusted models do not match the comparison input")
    out = model_dir / "M2_distribution_comparison"
    nb_dir = out / "ordinary_nb_laboratory"
    r_library = PROJECT_DIR / "03_discharge_note_text_analysis" / "01_language_complexity" / ".r-library"
    subprocess.run(["Rscript", "-", str(input_path), str(nb_dir), str(r_library), "M2_ordinary_laboratory"],
                   input=ADJUSTED_LONGITUDINAL_R, text=True, check=True)
    if hashlib.sha256(input_path.read_bytes()).hexdigest() != fingerprint:
        raise RuntimeError("Comparison inputs changed during fitting")
    ordinary = pd.read_csv(nb_dir / "adjusted_longitudinal_cohort_effects.csv")
    ordinary_diag = pd.read_csv(nb_dir / "adjusted_longitudinal_diagnostics.csv")
    if len(ordinary) != 10 or len(ordinary_diag) != 10:
        raise RuntimeError("Expected ten ordinary M2 laboratory fits")
    saved_effects = pd.read_csv(adjusted / "optimizer_checked_cohort_effects.csv")
    saved_diag = pd.read_csv(adjusted / "optimizer_checked_diagnostics.csv")
    zi_all = saved_effects[saved_effects.model.eq("M2_plus_prior_utilization")].copy()
    zi_diag_all = saved_diag[saved_diag.model.eq("M2_plus_prior_utilization")].copy()
    labs = ["laboratory_measurements", "laboratory_specimens"]
    ordinary_all = pd.concat([ordinary, zi_all[~zi_all.outcome.isin(labs)]], ignore_index=True)
    for frame in [ordinary_all, zi_all]:
        frame["FDR"] = float("nan")
        valid = frame.index[frame.converged & np.isfinite(frame.p_value)]
        values = frame.loc[valid, "p_value"].sort_values()
        adjusted_p = (values * 20 / pd.Series(range(1, len(values)+1), index=values.index)).iloc[::-1].cummin().iloc[::-1].clip(upper=1)
        frame.loc[adjusted_p.index, "FDR"] = adjusted_p
    ordinary_all.to_csv(out / "M2_ordinary_family_cohort_effects.csv", index=False)
    zi_all.to_csv(out / "M2_ZI_laboratory_family_cohort_effects.csv", index=False)
    compare_cols = ["cc_group", "outcome", "n_intervals", "n_admissions", "n_pairs", "rate_ratio", "ci_low", "ci_high", "p_value", "FDR", "converged"]
    comparison = ordinary_all[ordinary_all.outcome.isin(labs)][compare_cols].merge(
        zi_all[zi_all.outcome.isin(labs)][compare_cols], on=["cc_group", "outcome"], suffixes=("_NB", "_ZI"), validate="one_to_one")
    diag_cols = ["cc_group", "outcome", "AIC", "zero_screen_flag", "variance_screen_flag",
                 "observed_zero_fraction", "conditional_expected_zero_fraction", "adjacent_pearson_correlation"]
    dc = ordinary_diag[diag_cols].merge(zi_diag_all[zi_diag_all.outcome.isin(labs)][diag_cols],
        on=["cc_group", "outcome"], suffixes=("_NB", "_ZI"), validate="one_to_one")
    comparison = comparison.merge(dc, on=["cc_group", "outcome"], validate="one_to_one")
    comparison["AIC_change_ZI_minus_NB"] = comparison.AIC_ZI - comparison.AIC_NB
    comparison.loc[~(comparison.converged_NB & comparison.converged_ZI), "AIC_change_ZI_minus_NB"] = float("nan")
    for field in ["n_intervals", "n_admissions", "n_pairs"]:
        if not comparison[field+"_NB"].eq(comparison[field+"_ZI"]).all():
            raise ValueError("NB/ZI comparisons have different observations")
    comparison.to_csv(out / "M2_NB_ZI_laboratory_comparison.csv", index=False)
    baseline = pd.read_csv(model_dir / "zero_inflated_laboratory_models" / "exploratory_20_outcome_family.csv")
    baseline["stage"] = "M0"
    zi_all["stage"] = "M2"
    report_columns = ["stage", "cc_group", "outcome", "n_admissions", "n_pairs", "rate_ratio", "ci_low", "ci_high", "p_value", "FDR", "converged", "singular"]
    reporting = pd.concat([baseline[report_columns], zi_all[report_columns]], ignore_index=True)
    reporting["reporting_status"] = reporting.converged.map({True: "fitted; review diagnostics", False: "uncertainty unreliable; inference withheld"})
    reporting.loc[~reporting.converged, ["ci_low", "ci_high", "p_value", "FDR"]] = float("nan")
    reporting.to_csv(out / "M0_M2_main_model_reporting.csv", index=False)
    frames = pd.read_csv(input_path)
    zero_summary = frames.groupby(["pure_cc_group", "measure"]).n_events.agg(
        n_observed_intervals="size", n_zero=lambda y: int(y.eq(0).sum()), pct_zero=lambda y: 100*y.eq(0).mean()).reset_index()
    zero_summary.to_csv(out / "zero_event_intervals_by_CC.csv", index=False)
    frames.groupby("measure").n_events.agg(n_observed_intervals="size", n_zero=lambda y: int(y.eq(0).sum()),
        pct_zero=lambda y: 100*y.eq(0).mean()).reset_index().to_csv(out / "zero_event_intervals_overall.csv", index=False)
    (out / "model_input_sha256.txt").write_text(fingerprint + "\n")
    (out / "comparison_methods.txt").write_text(
        "M2 NB/ZI comparison uses identical laboratory observations, adjustments and random-effects structures.\n"
        "Ten new ordinary NB M2 lab fits compared with verified saved ZI M2 fits; original outputs preserved.\n"
        "Each alternative has a 20-comparison BH family including ten unchanged nonlaboratory M2 fits; nonconverged fits have no FDR.\n"
        "M0/M2 reporting uses ZI laboratories and ordinary NB microbiology/POE consistently at both stages.\n"
        "M0 is unadjusted for patient covariates, not for time, exposure or clustering.\n"
        "Zeros counted only in positive-exposure intervals: no post-discharge zeros or equal patient weighting.\n"
        "AIC change is interpreted only where both fits converge; simulation flags do not establish validity.\n")
    print(comparison[["cc_group", "outcome", "AIC_change_ZI_minus_NB", "rate_ratio_NB", "rate_ratio_ZI", "converged_NB", "converged_ZI"]].to_string(index=False))


def finalize_adjusted_comparison(output, model_dir, fingerprint):
    """Build checked reporting tables from saved estimates without refitting."""
    diagnostics = pd.read_csv(output / "adjusted_longitudinal_diagnostics.csv")
    estimates = pd.read_csv(output / "adjusted_longitudinal_cohort_effects.csv")
    baseline_dir = model_dir / "zero_inflated_laboratory_models"
    if (baseline_dir / "model_input_sha256.txt").read_text().strip() != fingerprint:
        raise ValueError("Zero-inflated baseline uses a different timing input")
    baseline = pd.read_csv(baseline_dir / "exploratory_20_outcome_family.csv")
    baseline["model"] = "M0_time_only"
    baseline["distribution"] = baseline.outcome.map(
        lambda value: "ZINB2" if value in ["laboratory_measurements", "laboratory_specimens"] else "NB2")
    compare_columns = ["model", "cc_group", "outcome", "distribution", "n_intervals", "n_admissions",
                       "n_pairs", "n_patients", "rate_ratio", "ci_low", "ci_high", "p_value", "FDR", "converged", "singular"]
    combined = pd.concat([baseline[compare_columns], estimates[compare_columns]], ignore_index=True)
    if combined.duplicated(["model", "cc_group", "outcome"]).any():
        raise ValueError("Duplicate adjusted comparison rows")
    for _, rows in combined.groupby(["cc_group", "outcome"]):
        if rows[["n_intervals", "n_admissions", "n_pairs", "n_patients"]].nunique().gt(1).any():
            raise ValueError("Admission or interval counts differ between adjustment stages")
    combined.to_csv(output / "M0_M4_cohort_effect_comparison.csv", index=False)
    diagnostics.groupby("model")[["converged", "singular", "zero_screen_flag", "variance_screen_flag"]].sum().to_csv(output / "diagnostic_flag_summary.csv")
    retry_path = output / "optimizer_review" / "same_model_optimizer_comparison.csv"
    if retry_path.exists():
        retries = pd.read_csv(retry_path)
        checked = estimates.copy()
        checked["optimizer"] = "nlminb"
        checked_diagnostics = diagnostics.copy()
        checked_diagnostics["optimizer"] = "nlminb"
        for _, retry in retries.loc[retries.eligible_retry].iterrows():
            mask = checked.model.eq(retry.model) & checked.cc_group.eq(retry.cc_group) & checked.outcome.eq(retry.outcome)
            diagnostic_mask = checked_diagnostics.model.eq(retry.model) & checked_diagnostics.cc_group.eq(retry.cc_group) & checked_diagnostics.outcome.eq(retry.outcome)
            if mask.sum() != 1 or diagnostic_mask.sum() != 1:
                raise ValueError("Optimizer retry cannot be linked to exactly one original model")
            for column in retry.index.intersection(checked.columns).difference(["model", "cc_group", "outcome"]):
                checked.loc[mask, column] = retry[column]
            for column in retry.index.intersection(checked_diagnostics.columns).difference(["model", "cc_group", "outcome"]):
                checked_diagnostics.loc[diagnostic_mask, column] = retry[column]
            checked.loc[mask, "optimizer"] = "BFGS_same_model_retry"
            checked_diagnostics.loc[diagnostic_mask, "optimizer"] = "BFGS_same_model_retry"
            checked_diagnostics.loc[diagnostic_mask, "status"] = "converged_after_optimizer_retry"
            checked_diagnostics.loc[diagnostic_mask, "optimizer_message"] = "Resolved by same-model BFGS retry; see optimizer_review"
        checked["FDR"] = float("nan")
        for _, rows in checked.groupby("model"):
            valid = rows.index[rows.converged & rows.p_value.notna()]
            values = checked.loc[valid, "p_value"].sort_values()
            adjusted = (values * 20 / pd.Series(range(1, len(values) + 1), index=values.index)).iloc[::-1].cummin().iloc[::-1].clip(upper=1)
            checked.loc[adjusted.index, "FDR"] = adjusted
        checked.to_csv(output / "optimizer_checked_cohort_effects.csv", index=False)
        checked_diagnostics.to_csv(output / "optimizer_checked_diagnostics.csv", index=False)
        checked_diagnostics.groupby("model")[["converged", "singular", "zero_screen_flag", "variance_screen_flag"]].sum().to_csv(output / "optimizer_checked_flag_summary.csv")
        baseline["optimizer"] = "original_baseline"
        pd.concat([baseline[compare_columns + ["optimizer"]], checked[compare_columns + ["optimizer"]]], ignore_index=True).to_csv(output / "optimizer_checked_M0_M4_comparison.csv", index=False)


LONGITUDINAL_REVIEW_R = r'''
args <- commandArgs(trailingOnly=TRUE)
.libPaths(c(args[4], .libPaths()))
suppressPackageStartupMessages(library(glmmTMB))
data <- read.csv(args[1],check.names=FALSE)
model_dir <- args[2]; out <- args[3]
dir.create(out,recursive=TRUE,showWarnings=FALSE)
data <- data[data$exposure_days > 0,]
data$mhc1 <- as.integer(data$cohort == "MHC1_psychotic")
data$patient <- factor(data$subject_id); data$matched_pair <- factor(data$pair_id)
data$admission <- factor(data$hadm_id)
data$period <- factor(data$interval,levels=c("0–24h","24–48h","48–72h","days 4–7","after day 7"))
stopifnot(!anyNA(data$period))
summary_rows <- list(); strata_rows <- list(); exposure_rows <- list(); re_rows <- list()
set.seed(20261007)
nsim <- 500
band <- function(x) as.numeric(quantile(x,c(.025,.5,.975)))
top_share <- function(y) {
  centered <- (y-mean(y))^2
  ix <- order(y,decreasing=TRUE)[seq_len(max(1,ceiling(length(y)*.01)))]
  if (sum(centered) == 0) return(NA_real_)
  sum(centered[ix])/sum(centered)
}
for (group in sort(unique(data$pure_cc_group))) for (measure in sort(unique(data$measure))) {
  key <- paste(group,measure,sep="__")
  cat("Review:",key,"\n"); flush.console()
  d <- droplevels(data[data$pure_cc_group == group & data$measure == measure,])
  fit <- readRDS(file.path(model_dir,paste0(gsub("[^a-zA-Z0-9_]","_",key),".rds")))
  frame <- model.frame(fit)
  stopifnot(identical(as.numeric(frame$n_events),as.numeric(d$n_events)),
    identical(as.character(frame$admission),as.character(d$admission)),
    identical(as.character(frame$period),as.character(d$period)),
    isTRUE(all.equal(as.numeric(model.offset(frame)),log(d$exposure_days))))
  y <- d$n_events; mu <- as.numeric(fitted(fit)); theta <- as.numeric(sigma(fit))
  pearson <- (y-mu)/sqrt(mu+mu^2/theta)
  predicted_zero <- dnbinom(0,size=theta,mu=mu)
  unconditional <- as.matrix(simulate(fit,nsim=nsim))
  # Plug-in conditional simulations retain fitted random effects. They are
  # explanatory diagnostics, not uncertainty-calibrated residual tests.
  conditional <- matrix(rnbinom(length(y)*nsim,size=theta,mu=rep(mu,nsim)),nrow=length(y))
  zb <- band(colMeans(unconditional==0)); vb <- band(apply(unconditional,2,var))
  qb <- band(apply(unconditional,2,quantile,probs=.99)); mb <- band(apply(unconditional,2,max))
  czb <- band(colMeans(conditional==0)); cvb <- band(apply(conditional,2,var))
  ord <- order(d$admission,d$period)
  same <- head(d$admission[ord],-1) == tail(d$admission[ord],-1)
  previous <- head(pearson[ord],-1)[same]; following <- tail(pearson[ord],-1)[same]
  vc <- VarCorr(fit)$cond
  summary_rows[[key]] <- data.frame(cc_group=group,outcome=measure,n_intervals=length(y),
    observed_zero_fraction=mean(y==0),conditional_expected_zero_fraction=mean(predicted_zero),
    unconditional_zero_low=zb[1],unconditional_zero_median=zb[2],unconditional_zero_high=zb[3],
    conditional_zero_low=czb[1],conditional_zero_high=czb[3],
    zero_screen_flag=mean(y==0)<zb[1] | mean(y==0)>zb[3],
    observed_variance=var(y),unconditional_variance_low=vb[1],unconditional_variance_high=vb[3],
    conditional_variance_low=cvb[1],conditional_variance_high=cvb[3],
    variance_screen_flag=var(y)<vb[1] | var(y)>vb[3],
    observed_q99=as.numeric(quantile(y,.99)),simulated_q99_low=qb[1],simulated_q99_high=qb[3],
    observed_max=max(y),simulated_max_low=mb[1],simulated_max_high=mb[3],
    top_one_percent_share_of_events=sum(sort(y,decreasing=TRUE)[seq_len(ceiling(length(y)*.01))])/sum(y),
    top_one_percent_share_of_squared_deviations=top_share(y),
    adjacent_pearson_correlation=cor(previous,following),
    adjacent_pearson_correlation_bounded=cor(pmax(-3,pmin(3,previous)),pmax(-3,pmin(3,following))))
  for (re in names(vc)) re_rows[[paste(key,re)]] <- data.frame(cc_group=group,outcome=measure,
    random_effect=re,variance=as.numeric(vc[[re]][1,1]),
    near_boundary=as.numeric(vc[[re]][1,1])<1e-6)
  for (cohort in unique(d$cohort)) for (period in levels(d$period)) {
    ix <- which(d$cohort==cohort & d$period==period)
    if (!length(ix)) next
    z <- band(colMeans(unconditional[ix,,drop=FALSE]==0))
    v <- band(apply(unconditional[ix,,drop=FALSE],2,var))
    cmean <- band(colMeans(unconditional[ix,,drop=FALSE]))
    strata_rows[[paste(key,cohort,period)]] <- data.frame(cc_group=group,outcome=measure,
      cohort=cohort,interval=period,n_intervals=length(ix),observed_days=sum(d$exposure_days[ix]),
      observed_mean=mean(y[ix]),conditional_fitted_mean=mean(mu[ix]),
      simulated_mean_low=cmean[1],simulated_mean_high=cmean[3],
      observed_zero_fraction=mean(y[ix]==0),conditional_expected_zero_fraction=mean(predicted_zero[ix]),
      simulated_zero_low=z[1],simulated_zero_high=z[3],
      observed_variance=var(y[ix]),simulated_variance_low=v[1],simulated_variance_high=v[3],
      mean_pearson_residual=mean(pearson[ix]),pearson_mean_square=mean(pearson[ix]^2),
      pooled_rate=sum(y[ix])/sum(d$exposure_days[ix]))
  }
  d$exposure_group <- cut(d$exposure_days*24,c(0,6,24,96,Inf),right=FALSE,
    labels=c("under 6h","6–24h","24–96h","96h or more"))
  for (period in levels(d$period)) for (exposure in levels(d$exposure_group)) {
    ix <- which(d$period==period & d$exposure_group==exposure)
    if (!length(ix)) next
    exposure_rows[[paste(key,period,exposure)]] <- data.frame(cc_group=group,outcome=measure,
      interval=period,exposure_group=exposure,n_intervals=length(ix),
      observed_zero_fraction=mean(y[ix]==0),conditional_expected_zero_fraction=mean(predicted_zero[ix]),
      observed_mean=mean(y[ix]),conditional_fitted_mean=mean(mu[ix]),
      mean_pearson_residual=mean(pearson[ix]),pearson_mean_square=mean(pearson[ix]^2))
  }
  # Static diagnostic panels: summaries only, no admission identifiers.
  png(file.path(out,paste0(gsub("[^a-zA-Z0-9_]","_",key),"_diagnostics.png")),width=1600,height=1000,res=160)
  par(mfrow=c(2,2),mar=c(4,4,3,1))
  s <- do.call(rbind,strata_rows)[do.call(rbind,strata_rows)$cc_group==group & do.call(rbind,strata_rows)$outcome==measure,]
  s <- s[order(match(s$cohort,c("MHC0","MHC1_psychotic")),match(s$interval,levels(d$period))),]
  cols <- ifelse(s$cohort=="MHC0","#238b45","#8055ad")
  labels <- paste(ifelse(s$cohort=="MHC0","C","P"),match(s$interval,levels(d$period)),sep="")
  x <- seq_len(nrow(s))
  plot(x,s$observed_zero_fraction,ylim=c(0,max(s$observed_zero_fraction,s$simulated_zero_high)*1.05+.01),
    pch=16,col=cols,xaxt="n",xlab="C=control, P=psychosis; period 1–5",ylab="Zero-event fraction",main="Observed zeros and 95% simulation envelopes")
  axis(1,x,labels); segments(x,s$simulated_zero_low,x,s$simulated_zero_high,col=cols)
  points(x,s$conditional_expected_zero_fraction,pch=4,col=cols)
  plot(x,s$mean_pearson_residual,pch=16,col=cols,xaxt="n",xlab="C=control, P=psychosis; period 1–5",ylab="Mean conditional Pearson residual",main="Residual pattern across time and cohort")
  axis(1,x,labels); abline(h=0,lty=2)
  plot(x,s$pearson_mean_square,pch=16,col=cols,xaxt="n",xlab="C=control, P=psychosis; period 1–5",ylab="Mean squared Pearson residual",main="Residual variability by interval")
  axis(1,x,labels); abline(h=1,lty=2)
  hist(log1p(y),breaks=35,col="grey80",main=paste(group,measure),xlab="log(1 + recorded count)")
  dev.off()
}
write.csv(do.call(rbind,summary_rows),file.path(out,"diagnostic_overview.csv"),row.names=FALSE)
write.csv(do.call(rbind,strata_rows),file.path(out,"diagnostics_by_cohort_interval.csv"),row.names=FALSE)
write.csv(do.call(rbind,exposure_rows),file.path(out,"diagnostics_by_exposure.csv"),row.names=FALSE)
write.csv(do.call(rbind,re_rows),file.path(out,"random_effect_boundary_review.csv"),row.names=FALSE)
writeLines(c(
  "Review of saved primary models only: no refitting, changed estimates or removed extreme observations.",
  "500 unconditional simulations redraw random effects; conditional plug-in draws hold fitted effects fixed.",
  "Simulation envelopes screen model-data compatibility, not formal p values or proof of zero inflation.",
  "Observed moments are compared with simulations under identical covariates/exposures; panels show conditional residual summaries.",
  "Conditional residual mean squares need not equal 1 after random effects are estimated; interpret descriptively.",
  "Adjacent residual correlations and bounded-residual checks are descriptive; no independent-observation significance tests.",
  "Random variances below 1e-6 are marked near-boundary, a deliberately broader screen than the initial 1e-8 flag.",
  "Exposure strata explore linear exposure-offset fit within time bins, not clinical comparisons of short/long stays.",
  "Multiple diagnostic screens are exploratory; sparse strata and simulation Monte Carlo variability require caution."
),file.path(out,"diagnostic_review_methods.txt"))
capture.output(sessionInfo(),file=file.path(out,"R_session_info.txt"))
'''


def review_longitudinal_models():
    """Diagnose saved models without reading raw records or altering estimates."""
    dataset = SCRIPT_DIR / "analysis_output_lab_linkage_and_timing" / "clinical_event_admission_intervals.csv"
    model_dir = PAIRED_WORKUP_OUTPUT_DIR / "longitudinal_negative_binomial_models"
    fingerprint = model_dir / "model_input_sha256.txt"
    if not fingerprint.exists() or fingerprint.read_text().strip() != hashlib.sha256(dataset.read_bytes()).hexdigest():
        raise ValueError("Saved longitudinal models do not match current interval inputs")
    output = model_dir / "diagnostic_review"
    r_library = PROJECT_DIR / "03_discharge_note_text_analysis" / "01_language_complexity" / ".r-library"
    # Feed the embedded program through stdin to avoid Rscript's -e size limit.
    subprocess.run(["Rscript", "-", str(dataset), str(model_dir), str(output), str(r_library)],
                   input=LONGITUDINAL_REVIEW_R, text=True, check=True)
    required = ["diagnostic_overview.csv", "diagnostics_by_cohort_interval.csv",
                "diagnostics_by_exposure.csv", "random_effect_boundary_review.csv"]
    if not all((output / name).exists() for name in required):
        raise RuntimeError("Longitudinal diagnostic review did not produce all expected outputs")


WORKUP_DIAGNOSTIC_R = r'''
args <- commandArgs(trailingOnly=TRUE)
.libPaths(c(args[3], .libPaths()))
suppressPackageStartupMessages(library(glmmTMB))
data <- read.csv(args[1]); out <- args[2]
dir.create(out, recursive=TRUE, showWarnings=FALSE)
data$mhc1 <- as.integer(data$cohort == "MHC1_psychotic")
data$patient <- factor(data$subject_id); data$matched_pair <- factor(data$pair_id)
set.seed(20261006)
rows <- list()
for (group in sort(unique(data$pure_cc_group))) {
  d <- droplevels(data[data$pure_cc_group == group, ])
  for (outcome in c("n_labevents_rows", "n_microbiologyevents_rows", "n_poe_rows")) {
    cat("Diagnostic comparisons:", group, outcome, "\n"); flush.console()
    # Every diagnostic subset retains both members of every included pair.
    valid <- with(d, is.finite(hospital_los_days) & hospital_los_days > 0)
    ids <- names(which(tapply(valid, d$matched_pair, all)))
    pos <- droplevels(d[d$matched_pair %in% ids, ])
    threshold <- quantile(d[[outcome]], .99)
    extreme_pairs <- d$matched_pair[d[[outcome]] > threshold]
    trimmed <- droplevels(d[!d$matched_pair %in% extreme_pairs, ])
    for (variant in c("NB2_M0", "NB1_M0", "NB2_zero_inflated", "NB2_age_elixhauser",
                      "NB2_positive_LOS_reference", "NB2_log_LOS_predictor", "NB2_tail_influence")) {
      dd <- if (variant == "NB2_tail_influence") trimmed else if (variant %in% c("NB2_positive_LOS_reference", "NB2_log_LOS_predictor")) pos else d
      stopifnot(all(table(dd$matched_pair) == 2))
      extra <- if (variant == "NB2_age_elixhauser") " + age_at_admission + elixhauser_score" else if (variant == "NB2_log_LOS_predictor") " + log(hospital_los_days)" else ""
      form <- as.formula(paste0(outcome, " ~ mhc1", extra, " + (1|patient) + (1|matched_pair)"))
      warn <- character()
      fit <- tryCatch(withCallingHandlers(
        glmmTMB(form, data=dd, family=if (variant == "NB1_M0") nbinom1 else nbinom2,
          ziformula=if (variant == "NB2_zero_inflated") ~1 else ~0,
          control=glmmTMBControl(optCtrl=list(iter.max=10000, eval.max=10000))),
        warning=function(w) { warn <<- c(warn, conditionMessage(w)); invokeRestart("muffleWarning") }), error=function(e) e)
      key <- paste(group,outcome,variant,sep="__")
      if (inherits(fit,"error")) {
        rows[[key]] <- data.frame(cc_group=group,outcome=outcome,variant=variant,converged=FALSE,message=conditionMessage(fit))
        next
      }
      converged <- fit$fit$convergence == 0 && isTRUE(fit$sdr$pdHess)
      cf <- summary(fit)$coefficients$cond
      sims <- as.matrix(simulate(fit, nsim=1000))
      zeros <- colMeans(sims == 0); variances <- apply(sims,2,var)
      zl <- quantile(zeros,.025); zh <- quantile(zeros,.975)
      vl <- quantile(variances,.025); vh <- quantile(variances,.975)
      rows[[key]] <- data.frame(cc_group=group,outcome=outcome,variant=variant,
        n_admissions=nrow(dd),n_pairs=nlevels(dd$matched_pair),converged=converged,
        AIC=if (converged) AIC(fit) else NA_real_,
        conditional_count_ratio=exp(cf["mhc1","Estimate"]),
        observed_zero_fraction=mean(dd[[outcome]] == 0),
        simulated_zero_low=zl,simulated_zero_high=zh,
        zero_screen_flag=mean(dd[[outcome]]==0)<zl || mean(dd[[outcome]]==0)>zh,
        observed_count_variance=var(dd[[outcome]]),
        simulated_variance_low=vl,simulated_variance_high=vh,
        variance_screen_flag=var(dd[[outcome]])<vl || var(dd[[outcome]])>vh,
        message=paste(unique(warn),collapse=" | "))
    }
  }
}
cols <- unique(unlist(lapply(rows,names)))
r <- do.call(rbind,lapply(rows,function(x) { x[setdiff(cols,names(x))] <- NA; x[cols] }))
write.csv(r,file.path(out,"diagnostic_model_comparisons.csv"),row.names=FALSE)
writeLines(c("Diagnostic experiments only: primary models and cohort are not changed.",
 "All models retain patient and original-pair intercepts. Subsets always remove complete pairs.",
 "AIC comparisons require identical admissions/outcomes; do not compare tail-trimmed AIC with full-sample AIC.",
 "LOS is a fitted log-duration predictor, NOT an offset; this changes the estimand and is not an automatic adjustment.",
 "Top 1% count-pair exclusion is an influence diagnostic, NOT a recommended exclusion rule.",
 "Zero-inflation is intercept-only. Conditional count ratios are diagnostic, not confirmatory estimates.",
 "Screening intervals use 1000 unconditional draws without refitting or parameter uncertainty; not formal residual-test p values."
),file.path(out,"diagnostic_notes.txt"))
'''


def diagnose_negative_binomial_models():
    """Audit counts and compare diagnostic models without changing primary fits."""
    import duckdb

    dataset = PAIRED_WORKUP_OUTPUT_DIR / "complete_cc_pair_workup_dataset.csv"
    data = pd.read_csv(dataset)
    out = PAIRED_WORKUP_OUTPUT_DIR / "negative_binomial_diagnostics_review"
    out.mkdir(exist_ok=True)
    rows = []
    for (group, cohort), frame in data.groupby(["pure_cc_group", "cohort"]):
        for outcome in WORKUP_COUNT_COLUMNS:
            y = frame[outcome]
            sq = (y - y.mean()) ** 2
            rows.append(dict(cc_group=group, cohort=cohort, outcome=outcome,
                n_admissions=len(frame), n_zero=int(y.eq(0).sum()),
                median=y.median(), q99=y.quantile(.99), maximum=y.max(),
                count_LOS_spearman=y.corr(frame.hospital_los_days, method="spearman"),
                top_1pct_squared_deviation_share=sq[y.gt(y.quantile(.99))].sum()/sq.sum(),
                zero_count_median_LOS=frame.loc[y.eq(0), "hospital_los_days"].median(),
                nonzero_count_median_LOS=frame.loc[y.gt(0), "hospital_los_days"].median()))
    pd.DataFrame(rows).to_csv(out / "count_distribution_audit.csv", index=False)
    # Aggregate-only, read-only database checks: no laboratory values or text.
    if common.DB_PATH.exists():
        with duckdb.connect(str(common.DB_PATH), read_only=True) as con:
            export_counts = data[ID_COLUMNS + WORKUP_COUNT_COLUMNS].copy()
            if "n_labevents_rows_original" in data.columns:
                export_counts["n_labevents_rows"] = data.n_labevents_rows_original
            con.register("retained_counts", export_counts)
            tables = set(con.execute("SHOW TABLES").fetchdf().iloc[:, 0])
            checks = []
            for event in ["labevents", "microbiologyevents", "poe", "poe_detail"]:
                table = common.EXPORT_BASENAMES[event]
                if table not in tables:
                    checks.append(dict(event=event, source_available=False))
                    continue
                result = con.execute(f"""
                    SELECT count(*) AS n_admissions_checked,
                      sum(r.n_{event}_rows != coalesce(s.n,0)) AS n_count_mismatches
                    FROM retained_counts r LEFT JOIN (
                      SELECT cohort,subject_id,hadm_id,count(*) AS n FROM {table}
                      GROUP BY cohort,subject_id,hadm_id
                    ) s USING(cohort,subject_id,hadm_id)
                """).fetchdf().iloc[0].to_dict()
                checks.append(dict(event=event, source_available=True, **result))
            pd.DataFrame(checks).to_csv(out / "source_count_validation.csv", index=False)
            # Investigate the documented possibility of NULL hadm_id labs during
            # a hospital stay; do not change attribution/counts automatically.
            if "labevents" in tables:
                desc = load_descriptors()[ID_COLUMNS + ["admittime", "dischtime"]]
                zero = data.loc[data.n_labevents_rows.eq(0), ID_COLUMNS + ["pure_cc_group"]].merge(desc, on=ID_COLUMNS, validate="one_to_one")
                con.register("zero_lab_admissions", zero)
                result = con.execute("""
                    WITH c AS (
                      SELECT z.cohort,z.pure_cc_group,z.hadm_id,count(l.subject_id) AS n
                      FROM zero_lab_admissions z LEFT JOIN labevents l
                        ON z.subject_id=l.subject_id AND l.hadm_id IS NULL
                        AND l.charttime BETWEEN z.admittime AND z.dischtime
                      GROUP BY z.cohort,z.pure_cc_group,z.hadm_id
                    ) SELECT cohort,pure_cc_group,count(*) AS n_zero_lab_admissions,
                      sum(n>0) AS n_with_unlinked_inpatient_labs,sum(n) AS n_unlinked_lab_rows
                      FROM c GROUP BY cohort,pure_cc_group
                """).fetchdf()
                result.to_csv(out / "zero_labs_unlinked_record_audit.csv", index=False)
    r_library = PROJECT_DIR / "03_discharge_note_text_analysis" / "01_language_complexity" / ".r-library"
    subprocess.run(["Rscript", "-e", WORKUP_DIAGNOSTIC_R, str(dataset), str(out), str(r_library)], check=True)

STAY_MORTALITY_R = r'''
args <- commandArgs(trailingOnly=TRUE)
.libPaths(c(args[3],.libPaths()))
suppressPackageStartupMessages(library(sandwich))
data <- read.csv(args[1]); out <- args[2]
data$patient <- factor(data$subject_id)
data$matched_pair <- factor(data$pair_id)
data$mhc1 <- as.integer(data$cohort=="MHC1_psychotic")
effects <- list(); coefficients <- list(); diagnostics <- list(); calibration <- list(); influence <- list()
models <- list(M0=character(),M2=c("age_at_admission_per_10y",
 "elixhauser_score_per_5pt","log1p_prior_all_admissions"))
if(length(args)>=4 && args[4]=="language_race_sensitivity") {
 data$language_group <- factor(data$language_group,
  levels=c("English","Non-English","Missing"))
 data$race_ethnicity_group <- factor(data$race_ethnicity_group,
  levels=c("White","Black","Asian","Hispanic/Latino",
           "Other recorded categories","Unknown/declined/missing"))
 models <- list(M3=c(models$M2,"language_group"),
                M4=c(models$M2,"language_group","race_ethnicity_group"))
}
for(stage in names(models)) for(group in sort(unique(data$pure_cc_group)))
 for(outcome in sort(unique(data$outcome))) {
 key <- paste(stage,group,outcome,sep="__")
 d <- droplevels(data[data$pure_cc_group==group & data$outcome==outcome,])
 stopifnot(all(table(d$matched_pair)==2),!anyNA(d),
  all(tapply(d$mhc1,d$matched_pair,sum)==1))
 mortality <- outcome %in% c("in_hospital_mortality","post_discharge_death_within_1y")
 form <- reformulate(c("mhc1",models[[stage]]),response="value")
 warnings <- character()
 fit <- withCallingHandlers(glm(form,data=d,
  family=if(mortality)binomial("logit") else quasipoisson("log"),
  control=glm.control(maxit=100,epsilon=1e-10),na.action=na.fail),
  warning=function(w){warnings <<- c(warnings,conditionMessage(w));invokeRestart("muffleWarning")})
 x <- model.matrix(fit)
 v <- tryCatch(vcovCL(fit,cluster=d[,c("patient","matched_pair")],
  type="HC1",cadjust=TRUE,multi0=FALSE,fix=FALSE),error=function(e)NULL)
 valid <- isTRUE(fit$converged) && fit$rank==ncol(x) && all(is.finite(coef(fit))) &&
  !is.null(v) && all(is.finite(v)) && all(diag(v)>0)
 mineig <- if(!is.null(v) && all(is.finite(v)))min(eigen(v,symmetric=TRUE,only.values=TRUE)$values) else NA_real_
 if(valid)valid <- mineig>= -1e-8*max(abs(eigen(v,symmetric=TRUE,only.values=TRUE)$values))
 # Working-residual scores work for both binomial/logit and quasi-Poisson/log.
 verification <- NA_real_
 if(valid){
  bread <- solve(crossprod(x,x*as.numeric(fit$weights)))
  scores <- x*as.numeric(fit$residuals*fit$weights)
  component <- function(g){s<-rowsum(scores,g,reorder=FALSE);G<-nrow(s)
   crossprod(s)*G/(G-1)*(nrow(x)-1)/(nrow(x)-ncol(x))}
  # Only observed intersections are needed; factor interaction can construct
  # millions of unused Cartesian-product levels in a whole-cohort run.
  intersection <- paste(as.character(d$patient),as.character(d$matched_pair),sep=":")
  manual <- bread %*% (component(d$patient)+component(d$matched_pair)-
   component(intersection)) %*% bread
  verification <- max(abs(v-manual))
  stopifnot(isTRUE(all.equal(unname(v),unname(manual),tolerance=1e-7)))
 }
 sparse <- mortality && sum(d$value)<20
 unstable <- mortality && (any(fitted(fit)<1e-10 | fitted(fit)>1-1e-10) || max(abs(coef(fit)))>20)
 valid <- valid && !unstable && !length(warnings)
 se <- if(valid)sqrt(diag(v)) else rep(NA_real_,length(coef(fit)))
 co <- data.frame(stage=stage,pure_cc_group=group,outcome=outcome,
  term=names(coef(fit)),beta=unname(coef(fit)),se=se)
 co$ratio <- exp(co$beta); co$ci_low <- exp(co$beta-1.96*co$se)
 co$ci_high <- exp(co$beta+1.96*co$se); co$p <- 2*pnorm(-abs(co$beta/co$se))
 co$effect_type <- if(mortality)"odds_ratio" else "mean_duration_ratio"
 co$numeric_valid <- valid; co$sparse_mortality <- sparse
 coefficients[[key]] <- co; effects[[key]] <- co[co$term=="mhc1",]
 # Rank approximate coefficient influence by summed estimating-equation scores.
 # Diagnostic deletions preserve complete pairs; primary fits retain all admissions.
 if(valid){
  for(kind in c("patient","matched_pair")){
   summed <- rowsum(scores,d[[kind]],reorder=FALSE)
   approximate <- as.numeric(summed %*% bread[,"mhc1"])
   selected <- order(abs(approximate),decreasing=TRUE)[seq_len(min(3,nrow(summed)))]
   for(rank in seq_along(selected)){
    cluster <- rownames(summed)[selected[rank]]
    affected_pairs <- unique(d$matched_pair[as.character(d[[kind]])==cluster])
    dd <- droplevels(d[!d$matched_pair %in% affected_pairs,])
    refit_warnings <- character(); refit_error <- ""
    refit <- tryCatch(withCallingHandlers(glm(form,data=dd,
     family=if(mortality)binomial("logit") else quasipoisson("log"),
     control=glm.control(maxit=100,epsilon=1e-10),na.action=na.fail),
     warning=function(w){refit_warnings <<- c(refit_warnings,conditionMessage(w));invokeRestart("muffleWarning")}),
     error=function(e){refit_error <<- conditionMessage(e);NULL})
    refit_beta <- if(is.null(refit))NA_real_ else unname(coef(refit)["mhc1"])
    accepted <- !is.null(refit) && isTRUE(refit$converged) &&
      all(is.finite(coef(refit))) && max(abs(coef(refit)))<20 && !length(refit_warnings)
    influence[[paste(key,kind,rank)]] <- data.frame(stage=stage,pure_cc_group=group,outcome=outcome,
     cluster_type=kind,score_rank=rank,n_removed_pairs=length(affected_pairs),
     original_beta=unname(coef(fit)["mhc1"]),refit_beta=refit_beta,
     absolute_beta_change=abs(refit_beta-unname(coef(fit)["mhc1"])),
     original_ratio=exp(coef(fit)["mhc1"]),refit_ratio=exp(refit_beta),
     refit_accepted=accepted,warnings=paste(refit_warnings,collapse="; "),error=refit_error)
   }
  }
 }
 diagnostics[[key]] <- data.frame(stage=stage,pure_cc_group=group,outcome=outcome,
  n_admissions=nrow(d),n_pairs=nlevels(d$matched_pair),n_patients=nlevels(d$patient),
  n_deaths=if(mortality)sum(d$value) else NA,converged=fit$converged,
  full_rank=fit$rank==ncol(x),numeric_valid=valid,min_covariance_eigenvalue=mineig,
  covariance_verification_error=verification,max_leverage=max(hatvalues(fit)),
  sparse_mortality=sparse,unstable_binary_fit=unstable,warnings=paste(warnings,collapse="; "))
 # Calibration by fitted-value quintile; descriptive, not a pass/fail test.
 d$prediction <- fitted(fit)
 d$prediction_bin <- pmin(5,ceiling(rank(d$prediction,ties.method="first")/nrow(d)*5))
 cal <- aggregate(cbind(value,prediction)~prediction_bin,data=d,FUN=mean)
 cal$stage <- stage; cal$pure_cc_group <- group; cal$outcome <- outcome
 calibration[[key]] <- cal
 saveRDS(list(fit=fit,cluster_covariance=v),file.path(out,paste0(gsub("[^A-Za-z0-9_]","_",key),".rds")))
 cat(key,"valid:",valid,"sparse mortality:",sparse,"\n")
}
for(item in c("effects","coefficients","diagnostics","calibration","influence"))
 write.csv(do.call(rbind,get(item)),file.path(out,paste0(item,".csv")),row.names=FALSE)
writeLines(capture.output(sessionInfo()),file.path(out,"R_session_info.txt"))
'''


def fit_whole_cohort_stay_mortality_sensitivity():
    """Add language/race to saved whole-cohort inputs without changing M0/M2."""
    import numpy as np
    from statsmodels.stats.multitest import multipletests

    baseline = SCRIPT_DIR / "analysis_output_whole_matched_stay_mortality"
    source = baseline / "model_inputs.csv"
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if source_hash != (baseline / "model_input_sha256.txt").read_text().strip():
        raise ValueError("Whole-cohort M0/M2 input fingerprint mismatch")
    output = baseline / "language_race_sensitivity"
    cov_path = (PROJECT_DIR / "03_discharge_note_text_analysis" / "01_language_complexity"
                / "analysis_output_language_complexity_inference" / "race_ethnicity_sensitivity"
                / "model_covariates.csv")
    numeric = ["age_at_admission_per_10y", "elixhauser_score_per_5pt", "log1p_prior_all_admissions"]
    categories = ["language_group", "race_ethnicity_group"]
    cov = pd.read_csv(cov_path, usecols=ID_COLUMNS + ["pair_id"] + numeric + categories)
    if cov.duplicated(ID_COLUMNS).any():
        raise ValueError("Duplicate admission keys in sensitivity covariates")
    data = pd.read_csv(source).merge(cov, on=ID_COLUMNS, how="left", validate="many_to_one",
                                   suffixes=("", "_source"), indicator=True)
    if not data._merge.eq("both").all() or not data.pair_id.eq(data.pair_id_source).all():
        raise ValueError("Missing/mismatched sensitivity admission or pair metadata")
    for column in numeric:
        np.testing.assert_allclose(data[column], data[column + "_source"], rtol=0, atol=1e-12)
    levels = {"language_group": ["English", "Non-English", "Missing"],
              "race_ethnicity_group": ["White", "Black", "Asian", "Hispanic/Latino",
                                       "Other recorded categories", "Unknown/declined/missing"]}
    for column, allowed in levels.items():
        if not data[column].isin(allowed).all():
            raise ValueError(f"Missing/unexpected categories in {column}")
    data = data.drop(columns=["_merge", "pair_id_source"] + [c + "_source" for c in numeric])
    if data.isna().any().any() or data.duplicated(["outcome", "hadm_id"]).any():
        raise ValueError("Incomplete or duplicate sensitivity input")
    if not data.groupby(["outcome", "pair_id"]).size().eq(2).all():
        raise ValueError("Sensitivity inputs do not preserve complete pairs")
    baseline_diagnostics = pd.read_csv(baseline / "diagnostics.csv")
    for outcome, frame in data.groupby("outcome"):
        original = baseline_diagnostics.loc[
            baseline_diagnostics.stage.eq("M2") & baseline_diagnostics.outcome.eq(outcome)]
        if len(original) != 1 or frame.pair_id.nunique() != int(original.n_pairs.iloc[0]):
            raise ValueError("Sensitivity sample differs from the M2 sample")
    output.mkdir(parents=True, exist_ok=True)
    data.to_csv(output / "model_inputs.csv", index=False)
    # Aggregate support/missingness audits, overall and for each endpoint.
    summaries = []
    for column in categories:
        counts = data.groupby(["outcome", "cohort", column]).agg(
            n_admissions=("hadm_id", "size"), n_patients=("subject_id", "nunique")
        ).reset_index().rename(columns={column: "category"})
        counts["variable"] = column
        summaries.append(counts)
    pd.concat(summaries, ignore_index=True).to_csv(output / "category_support.csv", index=False)
    r_library = PROJECT_DIR / "03_discharge_note_text_analysis" / "01_language_complexity" / ".r-library"
    subprocess.run(["Rscript", "-", str(output / "model_inputs.csv"), str(output), str(r_library),
                    "language_race_sensitivity"], input=STAY_MORTALITY_R, text=True, check=True)
    effects = pd.read_csv(output / "effects.csv")
    if len(effects) != 8 or effects.duplicated(["stage", "outcome"]).any():
        raise ValueError("Expected four endpoints in each of M3 and M4")
    for _, family in effects.groupby("stage"):
        adjusted = multipletests(family.p.fillna(1), method="fdr_bh")[1]
        effects.loc[family.index, "FDR"] = np.where(family.p.notna(), adjusted, np.nan)
    effects["analysis_family"] = "whole_matched_4_tests_per_sensitivity_stage"
    effects.to_csv(output / "effects.csv", index=False)
    keys = ["stage", "pure_cc_group", "outcome"]
    diagnostics = pd.read_csv(output / "diagnostics.csv")
    reporting = effects.merge(diagnostics[keys + ["n_pairs", "n_admissions", "n_patients"]],
                              on=keys, validate="one_to_one")
    reporting.to_csv(output / "M3_M4_reporting.csv", index=False)
    base_effects = pd.read_csv(baseline / "M0_M2_reporting.csv")
    comparison = pd.concat([base_effects.loc[base_effects.stage.eq("M2")], reporting], ignore_index=True)
    comparison.to_csv(output / "M2_M3_M4_comparison.csv", index=False)
    labels = {"hospital_los_days": "Hospital LOS", "ed_los_hours": "ED stay",
              "in_hospital_mortality": "In-hospital mortality",
              "post_discharge_death_within_1y": "365-day post-discharge mortality"}
    lines = ["# Whole matched cohort: language and race sensitivity", "",
             "M2: age, Elixhauser, log(1+prior hospital admissions).",
             "M3: M2 + recorded language group. M4: M3 + recorded race/ethnicity group.",
             "Same endpoint-specific admissions as M0/M2; PPML durations and logistic mortality,",
             "with patient/pair two-way HC1 clustered standard errors.", "",
             "| Outcome | Pairs | M2 ratio [95% CI]; FDR | M3 ratio [95% CI]; FDR | M4 ratio [95% CI]; FDR |",
             "|---|---:|---|---|---|"]
    for outcome, label in labels.items():
        rows = comparison.loc[comparison.outcome.eq(outcome)].set_index("stage")
        cells = []
        for stage in ["M2", "M3", "M4"]:
            r = rows.loc[stage]
            p_text = "<0.001" if r.FDR < .001 else f"{r.FDR:.3f}"
            cells.append(f"{r.ratio:.2f} [{r.ci_low:.2f}, {r.ci_high:.2f}]; {p_text}"
                         if r.numeric_valid else "Unavailable: numerical diagnostic")
        lines.append(f"| {label} | {int(rows.loc['M2', 'n_pairs'])} | " + " | ".join(cells) + " |")
    lines.extend(["", "Duration ratios are arithmetic mean-duration ratios; mortality ratios are odds ratios.",
                  "BH FDR uses four endpoints within each stage, retained for comparison with M0/M2.",
                  "This is not thesis-wide multiplicity control; final family policy remains to be settled.",
                  "Raw p-values are retained in effects.csv and the comparison CSV.",
                  "English and White are reference categories. Missing language and unknown/declined/missing race",
                  "are explicit categories, not imputed known values; these are recorded administrative fields.",
                  "Numerical acceptance and targeted influence checks do not establish all modelling assumptions."])
    (output / "results_summary.md").write_text("\n".join(lines) + "\n")
    (output / "methods.txt").write_text("\n".join(lines[2:6] + lines[-8:]) + "\n")
    (output / "source_baseline_input_sha256.txt").write_text(source_hash + "\n")
    for name, path in [("model_input_sha256.txt", output / "model_inputs.csv"),
                       ("covariate_source_sha256.txt", cov_path),
                       ("python_source_sha256.txt", Path(__file__))]:
        (output / name).write_text(hashlib.sha256(path.read_bytes()).hexdigest() + "\n")
    (output / "analysis_source_sha256.txt").write_text(hashlib.sha256(STAY_MORTALITY_R.encode()).hexdigest() + "\n")
    if hashlib.sha256(source.read_bytes()).hexdigest() != source_hash:
        raise ValueError("Baseline input changed during sensitivity analysis")
    print(reporting[["stage", "outcome", "n_pairs", "ratio", "ci_low", "ci_high", "FDR", "numeric_valid"]].to_string(index=False))


def fit_stay_mortality_models(post_discharge_only: bool = False, whole_matched: bool = False):
    """Outcome-specific complete pairs; M0/M2, no trimming or raw text access."""
    import numpy as np
    from statsmodels.stats.multitest import multipletests

    if whole_matched and post_discharge_only:
        raise ValueError("Whole-cohort mode fits all four endpoints together")
    output = (SCRIPT_DIR / "analysis_output_whole_matched_stay_mortality" if whole_matched
              else PAIRED_WORKUP_OUTPUT_DIR / ("post_discharge_1y_mortality_models" if post_discharge_only else "stay_and_mortality_models"))
    output.mkdir(parents=True, exist_ok=True)
    if whole_matched:
        pairs = pd.read_csv(common.MATCHED_PAIRS_PATH, usecols=[
            "pair_id", "mhc0_subject_id", "mhc0_hadm_id", "mhc1_subject_id", "mhc1_hadm_id"])
        if pairs.pair_id.duplicated().any():
            raise ValueError("Duplicate source matched-pair IDs")
        selected = pd.concat([
            pd.DataFrame({"pair_id": pairs.pair_id, "cohort": cohort,
                          "subject_id": pairs[f"{prefix}_subject_id"],
                          "hadm_id": pairs[f"{prefix}_hadm_id"]})
            for cohort, prefix in [("MHC0", "mhc0"), ("MHC1_psychotic", "mhc1")]
        ], ignore_index=True)
        # Retain the existing engine's group column, but do not apply a CC filter.
        selected["pure_cc_group"] = "whole_matched_cohort"
        if selected.hadm_id.duplicated().any() or selected[ID_COLUMNS].isna().any().any():
            raise ValueError("Missing or reused admission IDs in whole matched cohort")
        if set(selected.loc[selected.cohort.eq("MHC0"), "subject_id"]) & set(
                selected.loc[selected.cohort.eq("MHC1_psychotic"), "subject_id"]):
            raise ValueError("Patients overlap between whole matched cohorts")
    else:
        selected = pd.read_csv(PAIRED_INPUT_DIR / "complete_cc_pair_admissions.csv")
    descriptors = load_descriptors()
    include_post_discharge = post_discharge_only or whole_matched
    if whole_matched and set(descriptors.hadm_id) != set(selected.hadm_id):
        raise ValueError("Descriptor admission IDs differ from current whole matching")
    if include_post_discharge:
        # DOD contains dates, not trustworthy times of day. Discharge-alive
        # records may have a recorded post-discharge death on that same date.
        dod = pd.to_datetime(descriptors["dod"], errors="raise").dt.normalize()
        discharge = pd.to_datetime(descriptors["dischtime"], errors="raise")
        admission = pd.to_datetime(descriptors["admittime"], errors="raise")
        days = (dod - discharge.dt.normalize()).dt.days
        eligible = (descriptors.hospital_expire_flag.eq(0) & discharge.notna()
                    & admission.notna() & discharge.gt(admission)
                    & (dod.isna() | days.ge(0)))
        descriptors["post_discharge_death_within_1y"] = days.between(0,365).astype(float).where(eligible)
        descriptors["post_discharge_inhospital_death"] = descriptors.hospital_expire_flag.eq(1)
        descriptors["post_discharge_inconsistent_dates"] = (
            discharge.isna() | admission.isna() | discharge.le(admission)
            | (descriptors.hospital_expire_flag.eq(0) & dod.notna() & days.lt(0)))
        descriptors["post_discharge_same_day_death"] = eligible & days.eq(0)
    cov_path = (PROJECT_DIR / "03_discharge_note_text_analysis" / "01_language_complexity"
                / "analysis_output_language_complexity_inference" / "race_ethnicity_sensitivity"
                / "model_covariates.csv")
    covariates = ["age_at_admission_per_10y", "elixhauser_score_per_5pt", "log1p_prior_all_admissions"]
    cov = pd.read_csv(cov_path, usecols=ID_COLUMNS + covariates)
    extra_columns = (["post_discharge_death_within_1y","post_discharge_inhospital_death",
                      "post_discharge_inconsistent_dates","post_discharge_same_day_death"] if include_post_discharge else [])
    data = selected[ID_COLUMNS + ["pair_id", "pure_cc_group"]].merge(
        descriptors[ID_COLUMNS + ["hospital_los_days", "ed_los_hours", "hospital_expire_flag"] + extra_columns],
        on=ID_COLUMNS, how="left", validate="one_to_one").merge(
        cov[ID_COLUMNS + covariates], on=ID_COLUMNS, how="left", validate="one_to_one")
    if not data.groupby("pair_id").size().eq(2).all():
        raise ValueError("Incomplete source matching pairs")
    inputs, qc, descriptives = [], [], []
    outcomes = ([("post_discharge_death_within_1y","post_discharge_death_within_1y")] if post_discharge_only else [("hospital_los_days", "hospital_los_days"),
                            ("ed_los_hours", "ed_los_hours"),
                            ("in_hospital_mortality", "hospital_expire_flag")])
    if whole_matched:
        outcomes.append(("post_discharge_death_within_1y", "post_discharge_death_within_1y"))
    if include_post_discharge:
        data.groupby(["pure_cc_group","cohort"]).agg(
            initial_admissions=("hadm_id","size"),hospital_deaths=("post_discharge_inhospital_death","sum"),
            inconsistent_dates=("post_discharge_inconsistent_dates","sum"),
            same_day_post_discharge_deaths=("post_discharge_same_day_death","sum")
        ).reset_index().to_csv(output / "endpoint_definition_qc.csv",index=False)
    for outcome, column in outcomes:
        values = pd.to_numeric(data[column], errors="raise")
        mortality = outcome in ["in_hospital_mortality","post_discharge_death_within_1y"]
        valid_outcome = values.isin([0, 1]) if mortality else (np.isfinite(values) & values.gt(0))
        valid_cov = np.isfinite(data[covariates]).all(axis=1)
        valid_pairs = data.assign(valid=valid_outcome & valid_cov).groupby("pair_id").valid.all()
        keep = data.pair_id.isin(valid_pairs.index[valid_pairs])
        d = data.loc[keep, ID_COLUMNS + ["pair_id", "pure_cc_group"] + covariates].copy()
        d["outcome"] = outcome; d["value"] = values.loc[keep]
        inputs.append(d)
        for group, g in data.groupby("pure_cc_group"):
            mask = data.index.isin(g.index)
            qc.append(dict(outcome=outcome,pure_cc_group=group,initial_pairs=g.pair_id.nunique(),
                           retained_pairs=data.loc[mask & keep,"pair_id"].nunique(),
                           invalid_or_missing_outcomes=int((mask & ~valid_outcome).sum()),
                           missing_covariates=int((mask & ~valid_cov).sum())))
        for (group, cohort), g in d.groupby(["pure_cc_group", "cohort"]):
            descriptives.append(dict(outcome=outcome,pure_cc_group=group,cohort=cohort,
                n_admissions=len(g),n_patients=g.subject_id.nunique(),mean=g.value.mean(),
                sd=g.value.std(),median=g.value.median(),q1=g.value.quantile(.25),
                q3=g.value.quantile(.75),maximum=g.value.max(),
                deaths=int(g.value.sum()) if mortality else None))
    input_path = output / "model_inputs.csv"
    pd.concat(inputs, ignore_index=True).to_csv(input_path,index=False)
    pd.DataFrame(qc).to_csv(output / "sample_selection.csv",index=False)
    pd.DataFrame(descriptives).to_csv(output / "descriptives.csv",index=False)
    r_library = PROJECT_DIR / "03_discharge_note_text_analysis" / "01_language_complexity" / ".r-library"
    subprocess.run(["Rscript", "-", str(input_path), str(output), str(r_library)],
                   input=STAY_MORTALITY_R, text=True, check=True)
    effects = pd.read_csv(output / "effects.csv")
    if len(effects) != (8 if whole_matched else 10 if post_discharge_only else 30):
        raise ValueError("Unexpected number of cohort comparisons")
    effects["FDR"] = np.nan
    for stage, group in effects.groupby("stage"):
        # Whole cohort: all four endpoints per stage. Existing CC families unchanged.
        # Unavailable tests remain in their family as p=1.
        adjusted = multipletests(group.p.fillna(1), method="fdr_bh")[1]
        effects.loc[group.index,"FDR"] = np.where(group.p.notna(),adjusted,np.nan)
    effects.to_csv(output / "effects.csv",index=False)
    (output / "methods.txt").write_text(
        ("Entire matched cohort, without CC restrictions; outcome-specific complete pairs.\n"
         if whole_matched else "Five retained pure-CC matched subgroups; outcome-specific complete pairs.\n") +
        "M0: cohort. M2: cohort + age/10 years + Elixhauser/5 points + log(1+prior admissions).\n"
        "Identical admissions across M0/M2 for each outcome. No extreme-case trimming.\n"
        "Hospital LOS days and ED LOS hours: log-link PPML estimating arithmetic mean-duration ratios,\n"
        "implemented with quasi-Poisson estimating equations for continuous durations; no exposure offset.\n"
        "Mortality: admission hospital_expire_flag (0/1), logistic regression; odds ratios, not risk ratios.\n"
        "Uncertainty: patient and matched-pair two-way HC1 cluster covariance, intersection subtraction,\n"
        "cluster-number corrections; no eigenvalue repair. Covariance independently verified.\n"
        "Nonpositive/missing durations excluded with their paired partner; missing ED time is not zero.\n"
        f"BH FDR across {4 if whole_matched else 5 if post_discharge_only else 15} planned cohort-effect tests separately for M0 and M2.\n"
        "Fewer than 20 total deaths is a descriptive sparse-data caution, not a validated adequacy cutoff.\n"
        "Invalid/unstable fits have no inferential p-values. Calibration quintiles are descriptive.\n"
        "LOS includes in-hospital deaths; it is not time-to-discharge among survivors.\n")
    (output / "analysis_source_sha256.txt").write_text(hashlib.sha256(STAY_MORTALITY_R.encode()).hexdigest()+"\n")
    if post_discharge_only:
        with (output / "methods.txt").open("w") as f:
            f.write("One-year post-discharge mortality only; no duration or in-hospital mortality models in this run.\n"
                    "Logistic regression with patient/pair two-way HC1 clustered covariance; intersection subtraction.\n"
                    "M0 cohort; M2 age/10, Elixhauser/5, log(1+prior admissions); identical complete pairs per stage.\n"
                    "Odds ratios with normal-Wald confidence intervals; covariance independently verified.\n"
                    "Endpoint: recorded DOD on discharge date through discharge date +365 days inclusive,\n"
                    "among admissions documented discharged alive; missing DOD is no recorded death in window.\n"
                    "Require valid positive admission/discharge chronology; exclude pre-discharge DOD in survivor records.\n"
                    "Keep complete survivor pairs only; one death may label multiple overlapping discharge windows.\n"
                    "MIMIC DOD censoring: one year after last hospital discharge; registry ascertainment limitations remain.\n"
                    "Separate 5-test family per stage for this newly added endpoint; prior 15-test results unchanged.\n")
    if whole_matched:
        with (output / "methods.txt").open("a") as f:
            f.write("Post-discharge endpoint: recorded DOD on discharge date through +365 days inclusive,\n"
                    "among admissions documented discharged alive, with valid positive chronology;\n"
                    "exclude survivor records with pre-discharge DOD; missing DOD means no recorded death in window.\n"
                    "Keep complete survivor pairs; one death can label multiple overlapping windows.\n"
                    "DOD censoring/registry ascertainment limitations remain.\n"
                    "Whole-cohort four-endpoint FDR families are separate from existing CC-specific analyses.\n")
        (output / "python_source_sha256.txt").write_text(hashlib.sha256(Path(__file__).read_bytes()).hexdigest()+"\n")
        (output / "covariate_source_sha256.txt").write_text(hashlib.sha256(cov_path.read_bytes()).hexdigest()+"\n")
    (output / "model_input_sha256.txt").write_text(hashlib.sha256(input_path.read_bytes()).hexdigest()+"\n")
    print(effects[["stage","pure_cc_group","outcome","ratio","ci_low","ci_high","FDR","sparse_mortality","numeric_valid"]].to_string(index=False))


STAY_MORTALITY_MIXED_R = r'''
args <- commandArgs(trailingOnly=TRUE)
.libPaths(c(args[3],.libPaths()))
suppressPackageStartupMessages(library(glmmTMB))
data <- read.csv(args[1]); out <- args[2]
stopifnot(!anyNA(data),!anyDuplicated(data[,c("outcome","hadm_id")]))
data$patient <- factor(data$subject_id); data$matched_pair <- factor(data$pair_id)
data$mhc1 <- as.integer(data$cohort=="MHC1_psychotic")
models <- list(M0=character(),M2=c("age_at_admission_per_10y",
 "elixhauser_score_per_5pt","log1p_prior_all_admissions"))
effects <- list(); coefficients <- list(); diagnostics <- list(); checks <- list()
for(stage in names(models)) for(group in sort(unique(data$pure_cc_group)))
 for(outcome in sort(unique(data$outcome))) {
 key <- paste(stage,group,outcome,sep="__")
 cat("Mixed model:",key,"\n"); flush.console()
 d <- droplevels(data[data$pure_cc_group==group & data$outcome==outcome,])
 stopifnot(all(table(d$matched_pair)==2),all(tapply(d$mhc1,d$matched_pair,sum)==1),
           all(tapply(as.character(d$pure_cc_group),d$matched_pair,function(z)length(unique(z)))==1))
 mortality <- outcome %in% c("in_hospital_mortality","post_discharge_death_within_1y")
 stopifnot(if(mortality)all(d$value %in% c(0,1)) else all(d$value>0))
 form <- as.formula(paste("value ~",paste(c("mhc1",models[[stage]],
                     "(1 | patient)","(1 | matched_pair)"),collapse=" + ")))
 warnings <- character(); error <- ""
 fit <- tryCatch(withCallingHandlers(glmmTMB(form,data=d,
  family=if(mortality)binomial("logit") else Gamma("log"),REML=FALSE,
  control=glmmTMBControl(optCtrl=list(iter.max=2000,eval.max=3000,
                                   rel.tol=1e-10,x.tol=1e-10)),na.action=na.fail),
  warning=function(w){warnings <<- c(warnings,conditionMessage(w));invokeRestart("muffleWarning")}),
  error=function(e){error <<- conditionMessage(e);NULL})
 valid <- FALSE; patient_sd <- pair_sd <- gradient <- NA_real_
 initial_beta_difference <- NA_real_; verification_error <- ""
 if(!is.null(fit)){
  initial <- fit
  saveRDS(initial,file.path(out,paste0("initial_nlminb__",gsub("[^A-Za-z0-9_]","_",key),".rds")))
  # Uniform optimizer verification: same model, started at the initial estimates.
  verified <- tryCatch(withCallingHandlers(update(initial,
   start=list(beta=fixef(initial)$cond,betadisp=fixef(initial)$disp,theta=getME(initial,"theta")),
   control=glmmTMBControl(optimizer=optim,optArgs=list(method="BFGS"),
                         optCtrl=list(maxit=2000,reltol=1e-12))),
   warning=function(w){warnings <<- c(warnings,conditionMessage(w));invokeRestart("muffleWarning")}),
   error=function(e){verification_error <<- conditionMessage(e);NULL})
  if(!is.null(verified)){
   initial_beta_difference <- max(abs(fixef(verified)$cond-fixef(initial)$cond))
   # Prefer a verified fit only if its objective and stationarity are no worse.
   initial_gradient <- max(abs(initial$obj$gr(initial$fit$par)))
   verified_gradient <- max(abs(verified$obj$gr(verified$fit$par)))
   if(verified$fit$convergence==0 && verified$sdr$pdHess &&
      is.finite(verified$fit$objective) && verified$fit$objective<=initial$fit$objective+1e-7 &&
      verified_gradient<=initial_gradient)fit<-verified
  }
 }
 convergence <- NA_integer_; pd_hessian <- FALSE; boundary <- FALSE; extreme_binary_parameters <- FALSE
 beta <- se <- p <- low <- high <- ratio <- NA_real_
 if(!is.null(fit)){
  convergence <- fit$fit$convergence; pd_hessian <- isTRUE(fit$sdr$pdHess)
  vv <- VarCorr(fit)$cond
  patient_sd <- unname(attr(vv$patient,"stddev")); pair_sd <- unname(attr(vv$matched_pair,"stddev"))
  boundary <- min(patient_sd,pair_sd)<1e-4
  gradient <- max(abs(fit$obj$gr(fit$fit$par)))
  tab <- summary(fit)$coefficients$cond
  valid <- convergence==0 && pd_hessian && all(is.finite(tab)) && gradient<1e-3
  extreme_binary_parameters <- mortality && (any(abs(tab[,1])>10) || max(patient_sd,pair_sd)>10)
  if(extreme_binary_parameters)valid <- FALSE
  co <- data.frame(stage=stage,pure_cc_group=group,outcome=outcome,term=rownames(tab),
                   beta=tab[,1],se=tab[,2],numeric_valid=valid)
  co$ratio <- exp(co$beta)
  co$ci_low <- if(valid)exp(co$beta-1.96*co$se) else NA_real_
  co$ci_high <- if(valid)exp(co$beta+1.96*co$se) else NA_real_
  co$p <- if(valid)2*pnorm(-abs(co$beta/co$se)) else NA_real_
  coefficients[[key]] <- co
  cc <- co[co$term=="mhc1",]; beta<-cc$beta;se<-cc$se;ratio<-cc$ratio
  low<-cc$ci_low;high<-cc$ci_high;p<-cc$p
  saveRDS(fit,file.path(out,paste0(gsub("[^A-Za-z0-9_]","_",key),".rds")))
  # Descriptive predictive checks, with newly drawn random effects, no refits.
  # They assess outcome/group summaries, not formal calibrated goodness-of-fit p-values.
  if(valid){
   sims <- as.matrix(simulate(fit,nsim=250,seed=901+length(effects)))
   for(cohort in c("MHC0","MHC1_psychotic")){
    idx <- d$cohort==cohort
    stats <- list(mean=function(z)mean(z),variance=function(z)var(z),
                  maximum=function(z)max(z),q90=function(z)unname(quantile(z,.9)))
    if(mortality)stats <- stats[c("mean","variance")]
    for(stat in names(stats)){
     fun<-stats[[stat]]; observed<-fun(d$value[idx])
     predicted<-apply(sims[idx,,drop=FALSE],2,fun)
     lo<-unname(quantile(predicted,.025));hi<-unname(quantile(predicted,.975))
     checks[[paste(key,cohort,stat)]]<-data.frame(stage=stage,pure_cc_group=group,
      outcome=outcome,cohort=cohort,statistic=stat,observed=observed,
      simulation_low=lo,simulation_high=hi,outside_simulation_interval=observed<lo || observed>hi)
    }
   }
  }
 }
 effects[[key]] <- data.frame(stage=stage,pure_cc_group=group,outcome=outcome,
  beta=beta,se=se,ratio=ratio,ci_low=low,ci_high=high,p=p,
  effect_type=if(mortality)"conditional_odds_ratio" else "conditional_mean_duration_ratio",
  numeric_valid=valid,boundary_fit=boundary,sparse_mortality=mortality && sum(d$value)<20)
 diagnostics[[key]] <- data.frame(stage=stage,pure_cc_group=group,outcome=outcome,
  n_admissions=nrow(d),n_pairs=nlevels(d$matched_pair),n_patients=nlevels(d$patient),
  n_deaths=if(mortality)sum(d$value) else NA,optimizer_code=convergence,
  positive_definite_hessian=pd_hessian,max_gradient=gradient,numeric_valid=valid,
  patient_sd=patient_sd,pair_sd=pair_sd,patient_variance=patient_sd^2,pair_variance=pair_sd^2,
  random_effect_scale=if(mortality)"log_odds" else "log_mean_duration",boundary_fit=boundary,
  extreme_binary_parameters=extreme_binary_parameters,
  optimizer_verification_beta_difference=initial_beta_difference,
  warnings=paste(unique(warnings),collapse="; "),error=error,verification_error=verification_error)
}
for(item in c("effects","coefficients","diagnostics","checks"))
 write.csv(do.call(rbind,get(item)),file.path(out,paste0(item,".csv")),row.names=FALSE)
writeLines(capture.output(sessionInfo()),file.path(out,"R_session_info.txt"))
'''


def fit_stay_mortality_mixed_models(post_discharge_only: bool = False):
    """Reuse the exact saved outcome-specific inputs; preserve clustered results."""
    import numpy as np
    from statsmodels.stats.multitest import multipletests

    source = PAIRED_WORKUP_OUTPUT_DIR / ("post_discharge_1y_mortality_models" if post_discharge_only else "stay_and_mortality_models")
    input_path = source / "model_inputs.csv"
    if not input_path.exists():
        raise FileNotFoundError("Run --fit-stay-mortality first to prepare the common inputs")
    fingerprint = hashlib.sha256(input_path.read_bytes()).hexdigest()
    if fingerprint != (source / "model_input_sha256.txt").read_text().strip():
        raise ValueError("Clustered model input fingerprint mismatch")
    output = PAIRED_WORKUP_OUTPUT_DIR / ("post_discharge_1y_mortality_mixed_models" if post_discharge_only else "stay_and_mortality_mixed_models")
    output.mkdir(parents=True, exist_ok=True)
    r_library = PROJECT_DIR / "03_discharge_note_text_analysis" / "01_language_complexity" / ".r-library"
    subprocess.run(["Rscript", "-", str(input_path), str(output), str(r_library)],
                   input=STAY_MORTALITY_MIXED_R, text=True, check=True)
    if hashlib.sha256(input_path.read_bytes()).hexdigest() != fingerprint:
        raise RuntimeError("Common inputs changed during fitting")
    effects = pd.read_csv(output / "effects.csv")
    if len(effects) != (10 if post_discharge_only else 30) or effects.duplicated(["stage","pure_cc_group","outcome"]).any():
        raise ValueError("Unexpected number of unique mixed-model comparisons")
    effects["FDR"] = np.nan
    for stage, group in effects.groupby("stage"):
        adjusted = multipletests(group.p.fillna(1), method="fdr_bh")[1]
        effects.loc[group.index,"FDR"] = np.where(group.p.notna(),adjusted,np.nan)
    # All fits may fail diagnostic acceptance; R then writes an empty checks file.
    try:
        checks = pd.read_csv(output / "checks.csv")
    except pd.errors.EmptyDataError:
        checks = pd.DataFrame(columns=["stage","pure_cc_group","outcome","outside_simulation_interval"])
    predictive_flags = checks.groupby(["stage","pure_cc_group","outcome"]).outside_simulation_interval.any()
    effects["predictive_check_flag"] = [bool(predictive_flags.get((r.stage,r.pure_cc_group,r.outcome),False))
                                         if r.numeric_valid else None for r in effects.itertuples()]
    effects.to_csv(output / "effects.csv",index=False)
    previous = pd.read_csv(source / "effects.csv")
    comparison = effects.merge(previous,on=["stage","pure_cc_group","outcome"],
                               suffixes=("_mixed","_clustered"),validate="one_to_one")
    comparison.to_csv(output / "mixed_vs_clustered.csv",index=False)
    for name in ["sample_selection.csv","descriptives.csv"]:
        pd.read_csv(source/name).to_csv(output/name,index=False)
    (output / "model_input_sha256.txt").write_text(fingerprint+"\n")
    (output / "analysis_source_sha256.txt").write_text(hashlib.sha256(STAY_MORTALITY_MIXED_R.encode()).hexdigest()+"\n")
    (output / "methods.txt").write_text(
        "Same outcome-specific complete pairs, covariates and admission-level endpoints as clustered models.\n"
        "M0 cohort; M2 additionally age/10, Elixhauser/5, log(1+prior admissions).\n"
        "Gamma/log generalized linear mixed models for hospital LOS days and ED LOS hours.\n"
        "Binomial/logit generalized linear mixed models for in-hospital mortality.\n"
        "Crossed Gaussian patient and matched-pair random intercepts; constant Gamma dispersion; no offset.\n"
        "Maximum likelihood, Laplace approximation via glmmTMB; normal-Wald fixed-effect inference.\n"
        "Uniform nlminb then warm-start BFGS verification for every model; no subgroup-specific settings.\n"
        "Use verification fit only with optimizer code 0, positive Hessian, no worse objective/gradient.\n"
        "Exponentiated effects are conditional mean-duration ratios / conditional odds ratios.\n"
        "Not directly equivalent to marginal logistic odds ratios (non-collapsibility).\n"
        "Numerical validity requires optimizer code 0, positive Hessian, finite estimates/SE, max gradient <1e-3.\n"
        "Binary fits with |fixed coefficient|>10 or random SD>10 on the logit scale are flagged extreme\n"
        "and not accepted for Wald inference; this is an operational diagnostic, not proof of separation.\n"
        "Random-effect SD <1e-4 (linear-predictor scale) marked boundary; not automatically a failed optimizer.\n"
        "Invalid-fit Wald CI/p/FDR suppressed; boundary fits separately flagged, not silently simplified.\n"
        f"BH across {5 if post_discharge_only else 15} planned comparisons separately per stage, unavailable tests treated as p=1.\n"
        "250 unconditional simulations per numerically valid model for descriptive predictive checks.\n"
        "Checks compare group mean, variance, max, q90 (binary: mean/variance) with simulated 95% intervals.\n"
        "Predictive flags are exploratory, correlated and unadjusted, not formal goodness-of-fit tests.\n"
        "No patient deletion, rematching, trimming, distribution search, or automatic intercept removal.\n"
        "Sparse mortality (<20 deaths) is a caution, not a validated adequacy threshold.\n"
        "Terminal-event/informative cluster-size assumptions remain; random effects do not solve confounding.\n")
    if post_discharge_only:
        with (output / "methods.txt").open("w") as f:
            f.write("One-year post-discharge mortality: logistic mixed models with crossed Gaussian patient/pair intercepts.\n"
                    "M0 cohort; M2 age/10, Elixhauser/5, log(1+prior admissions); identical complete survivor pairs.\n"
                    "Maximum likelihood with Laplace approximation; uniform nlminb and warm-start BFGS verification.\n"
                    "Conditional odds ratios; normal-Wald CI/p, BH FDR across five CC comparisons per stage.\n"
                    "Invalid comparisons retained in FDR family as p=1; their CI/p/FDR are not reported.\n"
                    "Acceptance requires code 0, positive Hessian, finite estimates/SE, max gradient <1e-3.\n"
                    "|Fixed coefficient|>10 or random SD>10 on logit scale marked extreme; inference withheld.\n"
                    "Random SD<1e-4 marked boundary separately; these operational flags do not prove separation.\n"
                    "250 unconditional predictive simulations for accepted fits; descriptive mean/variance checks, no formal fit p-test.\n"
                    "Endpoint and eligibility:\n")
            f.write((source / "methods.txt").read_text().split("Endpoint:",1)[1])
    print(effects[["stage","pure_cc_group","outcome","ratio","ci_low","ci_high","FDR",
                   "numeric_valid","boundary_fit","sparse_mortality","predictive_check_flag"]].to_string(index=False))


def fit_primary_stay_mortality_models(whole_matched: bool = False):
    """Primary clinical outcome inference: PPML/logistic with two-way clustering."""
    if whole_matched:
        fit_stay_mortality_models(whole_matched=True)
        output = SCRIPT_DIR / "analysis_output_whole_matched_stay_mortality"
        source_base = SCRIPT_DIR
        families = [(output.name, "whole_matched_4_tests")]
    else:
        fit_stay_mortality_models()
        fit_stay_mortality_models(post_discharge_only=True)
        output = PAIRED_WORKUP_OUTPUT_DIR / "primary_stay_mortality_results"
        source_base = PAIRED_WORKUP_OUTPUT_DIR
        families = [("stay_and_mortality_models", "LOS_ED_inhospital_15_tests"),
                    ("post_discharge_1y_mortality_models", "post_discharge_5_tests")]
    output.mkdir(parents=True, exist_ok=True)
    for name in ["effects.csv","coefficients.csv","diagnostics.csv","descriptives.csv",
                 "sample_selection.csv","calibration.csv","influence.csv"]:
        frames = []
        for folder, family in families:
            frame = pd.read_csv(source_base / folder / name)
            frame["analysis_family"] = family
            frames.append(frame)
        pd.concat(frames, ignore_index=True).to_csv(output / name,index=False)
    effects = pd.read_csv(output / "effects.csv")
    diagnostics = pd.read_csv(output / "diagnostics.csv")
    keys = ["stage","pure_cc_group","outcome"]
    expected = 8 if whole_matched else 40
    if len(effects)!=expected or effects.duplicated(keys).any():
        raise ValueError(f"Expected {expected} distinct cohort comparisons")
    report = effects.merge(diagnostics[keys+["n_pairs","n_admissions","n_patients"]],
                           on=keys,validate="one_to_one")
    report.to_csv(output / "M0_M2_reporting.csv",index=False)
    labels = {"hospital_los_days":"Hospital LOS","ed_los_hours":"ED stay",
              "in_hospital_mortality":"In-hospital mortality",
              "post_discharge_death_within_1y":"365-day post-discharge mortality"}
    def number(value):
        return "<0.001" if value<.001 else f"{value:.3f}"
    lines = ["# Primary hospital outcomes: " + ("whole matched cohort" if whole_matched else "CC-specific cohort comparisons"), "",
             "PPML with a log link for durations; logistic regression for both mortality endpoints.",
             "Patient and matched-pair two-way clustered HC1 standard errors for every model.",
             "M0: cohort. M2: additionally age, Elixhauser score, and log(1+prior hospital admissions).", "",
             "| Analysis population | Outcome | Pairs | M0 ratio [95% CI]; FDR | M2 ratio [95% CI]; FDR |",
             "|---|---|---:|---|---|"]
    for cc in sorted(report.pure_cc_group.unique()):
        for outcome, label in labels.items():
            rows = report[(report.pure_cc_group==cc)&(report.outcome==outcome)].set_index("stage")
            cells=[]
            for stage in ["M0","M2"]:
                r=rows.loc[stage]
                cells.append(f"{r.ratio:.2f} [{r.ci_low:.2f}, {r.ci_high:.2f}]; {number(r.FDR)}"
                             if r.numeric_valid else "Unavailable: numerical diagnostic")
            caution=" (sparse deaths)" if rows.sparse_mortality.any() else ""
            lines.append(f"| {cc} | {label}{caution} | {int(rows.loc['M0','n_pairs'])} | {cells[0]} | {cells[1]} |")
    lines.extend(["", "Duration ratios concern arithmetic means; mortality ratios are odds ratios.",
        ("BH FDR: four whole-cohort endpoints per stage, separate from all CC-specific tests."
         if whole_matched else "BH FDR: 15 LOS/ED/in-hospital comparisons per stage; five post-discharge comparisons separately."),
        "Families were specified before inspecting the new p-values.",
        "Fewer than 20 deaths is a descriptive caution, not a validated adequacy threshold.",
        "Convergence does not certify adequate inference when deaths are sparse or influential patients dominate.",
        "influence.csv checks three score-ranked patients and three pairs per fit; it is not exhaustive.",
        "Diagnostic removal of a patient also removes all affected pairs; no primary observations are removed.",
        "Post-discharge analyses use complete survivor pairs, with admission-specific 365-day windows.",
        "One death can label multiple overlapping windows. These are associations, not causal effects.",
        "The raw in-hospital endpoint is admission-based; clustering does not convert its denominator to unique patients.",
        "Mixed models are retained as exploratory alternatives; language-complexity models are a separate analysis."])
    (output / "results_summary.md").write_text("\n".join(lines)+"\n")
    fingerprints = {folder: (source_base/folder/"model_input_sha256.txt").read_text().strip()
                    for folder,_ in families}
    (output / "source_input_fingerprints.txt").write_text(
        "\n".join(f"{folder}: {fingerprint}" for folder,fingerprint in fingerprints.items())+"\n")
    print(f"Primary reporting saved: {len(effects)} comparisons; {int(effects.numeric_valid.sum())} numerically accepted.")


def main():
    selected = pd.read_csv(PAIRED_INPUT_DIR / "complete_cc_pair_admissions.csv")
    descriptors = load_descriptors()
    counts = load_or_build_utilization_counts()
    if descriptors.duplicated(ID_COLUMNS).any() or counts.duplicated(ID_COLUMNS).any():
        raise ValueError("Duplicate descriptor/utilization admission keys")
    for label, table in [("descriptors", descriptors), ("counts", counts)]:
        check = selected[ID_COLUMNS].merge(table[ID_COLUMNS], on=ID_COLUMNS, how="left", indicator=True, validate="one_to_one")
        if not check._merge.eq("both").all():
            raise ValueError(f"Missing retained admission rows in {label}")
    data = selected.merge(descriptors[ID_COLUMNS + ["hospital_los_days", "ed_los_hours"]],
                          on=ID_COLUMNS, how="left", validate="one_to_one")
    data = data.merge(counts[ID_COLUMNS + WORKUP_COUNT_COLUMNS], on=ID_COLUMNS, how="left", validate="one_to_one")
    correction_path = SCRIPT_DIR / "analysis_output_lab_linkage_and_timing" / "lab_linkage_corrected_workup_dataset.csv"
    if correction_path.exists():
        correction = pd.read_csv(correction_path)
        audit_columns = ["n_labevents_rows_original", "n_recovered_lab_rows", "n_inpatient_lab_rows",
                         "n_inpatient_lab_specimens", "inpatient_lab_rows_per_hospital_day"]
        correction = correction[ID_COLUMNS + ["pair_id", "pure_cc_group", "n_labevents_rows"] + audit_columns]
        joined = data.merge(correction, on=ID_COLUMNS, how="outer", validate="one_to_one",
                            suffixes=("", "_corrected"), indicator=True)
        if not joined._merge.eq("both").all():
            raise ValueError("Lab correction has a different admission set; rerun 06")
        if not joined.pair_id.eq(joined.pair_id_corrected).all() or not joined.pure_cc_group.eq(joined.pure_cc_group_corrected).all():
            raise ValueError("Lab correction has different pairing/CC labels; rerun 06")
        if not joined.n_labevents_rows.eq(joined.n_labevents_rows_original).all():
            raise ValueError("Lab source counts changed since recovery; rerun 06")
        if not (joined.n_labevents_rows_corrected == joined.n_labevents_rows_original + joined.n_recovered_lab_rows).all():
            raise ValueError("Invalid laboratory recovery arithmetic")
        joined["n_labevents_rows"] = joined.n_labevents_rows_corrected
        data = joined.drop(columns=["_merge", "pair_id_corrected", "pure_cc_group_corrected", "n_labevents_rows_corrected"])
        print(f"Applied validated laboratory recovery: {int(data.n_recovered_lab_rows.sum())} rows across {int(data.n_recovered_lab_rows.gt(0).sum())} admissions.")
    if not data.groupby("pair_id").size().eq(2).all():
        raise ValueError("Incomplete pair after joins")
    qc = []
    for measure in WORKUP_MEASURES:
        values = pd.to_numeric(data[measure], errors="raise")
        negative = values.lt(0)
        qc.append({"measure": measure, "n_missing_input": int(values.isna().sum()),
                   "n_negative_input": int(negative.sum())})
        if measure in WORKUP_COUNT_COLUMNS and (negative.any() or values.isna().any()):
            raise ValueError(f"Missing/negative recorded counts: {measure}")
        data[measure] = values.mask(negative)
    for count in WORKUP_COUNT_COLUMNS:
        data[count + "_per_hospital_day"] = data[count] / data.hospital_los_days.where(data.hospital_los_days.gt(0))
    measures = [*WORKUP_MEASURES, *(c + "_per_hospital_day" for c in WORKUP_COUNT_COLUMNS)]
    rows = []
    pair_rows = []
    for (group, cohort), frame in data.groupby(["pure_cc_group", "cohort"]):
        for measure in measures:
            group_data = data[data.pure_cc_group.eq(group)]
            valid_counts = group_data[group_data[measure].notna()].groupby("pair_id").size()
            complete_ids = valid_counts[valid_counts.eq(2)].index
            for sample in ["available_admissions", "complete_outcome_pairs"]:
                selected_frame = frame if sample == "available_admissions" else frame[frame.pair_id.isin(complete_ids)]
                values = selected_frame[measure].dropna()
                rows.append({"cc_group": group, "cohort": cohort, "measure": measure, "analysis_sample": sample,
                             "n_admissions": len(selected_frame), "n_nonmissing": len(values), "n_missing": selected_frame[measure].isna().sum(),
                             "mean": values.mean(), "sd": values.std(), "median": values.median(),
                             "q1": values.quantile(.25), "q3": values.quantile(.75),
                             "min": values.min(), "max": values.max(),
                             "n_zero": int(values.eq(0).sum()), "pct_zero": 100*values.eq(0).mean()})
    for group, frame in data.groupby("pure_cc_group"):
        for measure in measures:
            paired = frame.pivot(index="pair_id", columns="cohort", values=measure).dropna()
            difference = paired.MHC1_psychotic - paired.MHC0
            pair_rows.append({"cc_group": group, "measure": measure, "n_complete_outcome_pairs": len(paired),
                              "mean_paired_difference": difference.mean(), "median_paired_difference": difference.median(),
                              "q1_paired_difference": difference.quantile(.25), "q3_paired_difference": difference.quantile(.75)})
    PAIRED_WORKUP_OUTPUT_DIR.mkdir(exist_ok=True)
    data.to_csv(PAIRED_WORKUP_OUTPUT_DIR / "complete_cc_pair_workup_dataset.csv", index=False)
    model_dir = PAIRED_WORKUP_OUTPUT_DIR / "negative_binomial_models"
    if (model_dir / "negative_binomial_cohort_effects.csv").exists():
        fingerprint = model_dir / "model_input_sha256.txt"
        current_hash = hashlib.sha256((PAIRED_WORKUP_OUTPUT_DIR / "complete_cc_pair_workup_dataset.csv").read_bytes()).hexdigest()
        matching = fingerprint.exists() and fingerprint.read_text().strip() == current_hash
        status = "Model estimates match current inputs.\n" if matching else "STALE: existing model estimates predate or cannot be verified against the current dataset. Rerun modelling before reporting these estimates.\n"
        (model_dir / "model_input_status.txt").write_text(status)
        if not matching:
            print(status.strip())
    summary = pd.DataFrame(rows)
    summary.to_csv(PAIRED_WORKUP_OUTPUT_DIR / "complete_cc_pair_workup_summary.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(PAIRED_WORKUP_OUTPUT_DIR / "complete_cc_pair_workup_paired_differences.csv", index=False)
    pd.DataFrame(qc).to_csv(PAIRED_WORKUP_OUTPUT_DIR / "complete_cc_pair_workup_qc.csv", index=False)
    print(summary[summary.measure.isin(WORKUP_MEASURES) & summary.analysis_sample.eq("complete_outcome_pairs")][["cc_group", "cohort", "measure", "n_nonmissing", "n_missing", "mean", "median", "q1", "q3", "pct_zero"]].to_string(index=False))
    print("\nInput quality checks:")
    print(pd.DataFrame(qc).to_string(index=False))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-negative-binomial", action="store_true", help="Fit simple patient/pair NB2 count models after summaries")
    parser.add_argument("--model-only", action="store_true", help="Fit NB2 models from the existing work-up dataset without rebuilding summaries")
    parser.add_argument("--diagnose-negative-binomial", action="store_true", help="Audit counts and fit exploratory paired diagnostic variants; preserve primary results")
    parser.add_argument("--fit-longitudinal", action="store_true", help="Fit time-adjusted NB2 rate models from saved inpatient intervals only")
    parser.add_argument("--review-longitudinal", action="store_true", help="Review saved longitudinal models without changing primary estimates")
    parser.add_argument("--fit-longitudinal-zi", action="store_true", help="Fit constant-zero ZINB2 laboratory alternatives without changing original models")
    parser.add_argument("--fit-longitudinal-adjusted", action="store_true", help="Fit M1-M4 clinical models from existing covariates and reassess diagnostics")
    parser.add_argument("--compare-m2-distributions", action="store_true", help="Compare ordinary and zero-inflated M2 labs and retain M0/M2 reporting")
    parser.add_argument("--without-admission-intercept", action="store_true", help="Retest M0/M2 uniformly with patient and pair intercepts only")
    parser.add_argument("--fit-marginal", action="store_true", help="Fit M0/M2/M3/M4 PPML with patient/pair two-way cluster-robust inference")
    parser.add_argument("--fit-first24-marginal", action="store_true", help="Fit the same PPML stages using one first-24h observation per admission")
    parser.add_argument("--fit-first12-marginal", action="store_true", help="Recount first-12h events and fit the same PPML stages")
    parser.add_argument("--fit-stay-mortality", action="store_true", help="Fit M0/M2 LOS, ED stay and in-hospital mortality with patient/pair clustered inference")
    parser.add_argument("--fit-stay-mortality-mixed", action="store_true", help="Fit Gamma/log and logistic patient/pair mixed models using identical saved inputs")
    parser.add_argument("--fit-post-discharge-mortality", action="store_true", help="Fit primary M0/M2 post-discharge logistic models with patient/pair clustered inference")
    parser.add_argument("--fit-primary-stay-mortality", action="store_true", help="Fit and report all four primary clinical endpoints using patient/pair clustered inference")
    parser.add_argument("--fit-whole-cohort-stay-mortality", action="store_true", help="Fit M0/M2 for all four endpoints across the full matched cohort; save separate outputs")
    parser.add_argument("--fit-whole-cohort-stay-mortality-sensitivity", action="store_true", help="Fit M3 language and M4 language/race adjustments on the saved whole-cohort M0/M2 sample")
    options = parser.parse_args()
    if not options.model_only and not options.diagnose_negative_binomial and not options.fit_longitudinal and not options.review_longitudinal and not options.fit_longitudinal_zi and not options.fit_longitudinal_adjusted and not options.compare_m2_distributions and not options.without_admission_intercept and not options.fit_marginal and not options.fit_first24_marginal and not options.fit_first12_marginal and not options.fit_stay_mortality and not options.fit_stay_mortality_mixed and not options.fit_post_discharge_mortality and not options.fit_primary_stay_mortality and not options.fit_whole_cohort_stay_mortality and not options.fit_whole_cohort_stay_mortality_sensitivity:
        main()
    if options.fit_negative_binomial or options.model_only:
        fit_negative_binomial_models()
    if options.diagnose_negative_binomial:
        diagnose_negative_binomial_models()
    if options.fit_longitudinal:
        fit_longitudinal_models()
    if options.review_longitudinal:
        review_longitudinal_models()
    if options.fit_longitudinal_zi:
        fit_zero_inflated_longitudinal_models()
    if options.fit_longitudinal_adjusted:
        fit_adjusted_longitudinal_models()
    if options.compare_m2_distributions:
        fit_m2_distribution_comparison()
    if options.without_admission_intercept:
        fit_without_admission_intercept()
    if options.fit_marginal:
        fit_marginal_longitudinal_models()
    if options.fit_first24_marginal:
        fit_marginal_longitudinal_models(first24=True)
    if options.fit_first12_marginal:
        fit_marginal_longitudinal_models(first12=True)
    if options.fit_stay_mortality:
        fit_stay_mortality_models()
    if options.fit_stay_mortality_mixed:
        fit_stay_mortality_mixed_models()
    if options.fit_post_discharge_mortality:
        fit_stay_mortality_models(post_discharge_only=True)
    if options.fit_primary_stay_mortality:
        fit_primary_stay_mortality_models()
    if options.fit_whole_cohort_stay_mortality:
        fit_primary_stay_mortality_models(whole_matched=True)
    if options.fit_whole_cohort_stay_mortality_sensitivity:
        fit_whole_cohort_stay_mortality_sensitivity()
