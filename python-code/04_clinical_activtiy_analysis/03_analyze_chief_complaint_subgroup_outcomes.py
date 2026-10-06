"""Describe work-up and stay duration within five complete pure-CC pairs.

Run 01_describe_chief_complaint_subgroups.py first. The primary summaries retain
both members of an original matched pair with valid values for each outcome.
Missing/negative stay durations are excluded per outcome, without converting
missing event counts to zero. Event counts represent recorded database rows.
Optional negative-binomial inference uses crossed patient/pair random intercepts.
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
            "04_clinical_activtiy_analysis/01_describe_chief_complaint_subgroups.py first: "
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
    options = parser.parse_args()
    if not options.model_only and not options.diagnose_negative_binomial:
        main()
    if options.fit_negative_binomial or options.model_only:
        fit_negative_binomial_models()
    if options.diagnose_negative_binomial:
        diagnose_negative_binomial_models()
