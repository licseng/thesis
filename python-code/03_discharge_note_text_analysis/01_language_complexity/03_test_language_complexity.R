#!/usr/bin/env Rscript

# Inference for matched-cohort language-complexity outcomes.
#
# For each outcome this script fits a crossed-random-intercept linear model:
#
#   outcome ~ cohort + (1 | patient) + (1 | matched_pair)
#
# It also performs paired t-tests and Wilcoxon signed-rank tests using the
# original one-to-one admission matches. Three nested extensions of the mixed
# model adjust for age and Elixhauser score, prior hospital utilization, and
# recorded patient language.

args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)
if (length(file_arg) != 1) stop("Could not determine this script's path.")
script_path <- normalizePath(sub("^--file=", "", file_arg))
script_dir <- dirname(script_path)
python_dir <- dirname(dirname(script_dir))

local_library <- file.path(script_dir, ".r-library")
.libPaths(c(local_library, .libPaths()))

suppressPackageStartupMessages({
  library(lme4)
  library(lmerTest)
})

metrics_dir <- file.path(script_dir, "analysis_output_language_complexity")
full_note_path <- file.path(metrics_dir, "language_complexity_note_level_metrics.csv")
section_path <- file.path(metrics_dir, "language_complexity_prose_section_metrics.csv")
pairs_path <- file.path(
  python_dir,
  "02_cohort_matching",
  "matched_cohort_output",
  "matched_pairs.csv"
)
output_dir <- file.path(script_dir, "analysis_output_language_complexity_inference")
covariates_path <- file.path(output_dir, "language_complexity_model_covariates.csv")
race_sensitivity_only <- "--race-sensitivity" %in% commandArgs(trailingOnly=TRUE)
if (race_sensitivity_only) {
  output_dir <- file.path(output_dir, "race_ethnicity_sensitivity")
  covariates_path <- file.path(output_dir, "model_covariates.csv")
}

required_paths <- c(full_note_path, section_path, pairs_path, covariates_path)
missing_paths <- required_paths[!file.exists(required_paths)]
if (length(missing_paths) > 0) {
  stop(paste("Missing required inputs:", paste(missing_paths, collapse = "\n")))
}

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

full_note_outcomes <- c(
  "mean_words_per_sentence",
  "mean_words_per_line",
  "pct_short_lines",
  "mean_characters_per_alpha_word",
  "pct_long_alpha_words",
  "pct_complex_alpha_words",
  "flesch_reading_ease",
  "flesch_kincaid_grade",
  "gunning_fog_index",
  "smog_index",
  "automated_readability_index"
)

# Parsed prose sections have been whitespace-flattened. Consequently,
# mean_words_per_line and pct_short_lines are not valid section-level metrics.
section_outcomes <- setdiff(
  full_note_outcomes,
  c("mean_words_per_line", "pct_short_lines")
)

pairs <- read.csv(pairs_path, check.names = FALSE)
pair_map <- rbind(
  data.frame(
    pair_id = pairs$pair_id,
    cohort = "MHC1_psychotic",
    subject_id = pairs$mhc1_subject_id,
    hadm_id = pairs$mhc1_hadm_id
  ),
  data.frame(
    pair_id = pairs$pair_id,
    cohort = "MHC0",
    subject_id = pairs$mhc0_subject_id,
    hadm_id = pairs$mhc0_hadm_id
  )
)

attach_pair_ids <- function(metrics) {
  merged <- merge(
    metrics,
    pair_map,
    by = c("cohort", "subject_id", "hadm_id"),
    all.x = TRUE,
    all.y = FALSE,
    sort = FALSE
  )
  if (anyNA(merged$pair_id)) stop("Some metric rows could not be linked to a matched pair.")
  merged$cohort <- factor(merged$cohort, levels = c("MHC0", "MHC1_psychotic"))
  merged$patient_cluster <- factor(paste(merged$cohort, merged$subject_id, sep = "_"))
  merged$pair_cluster <- factor(merged$pair_id)
  merged
}

model_covariates <- read.csv(covariates_path, check.names = FALSE)
model_covariates$language_group <- factor(
  model_covariates$language_group,
  levels = c("English", "Non-English", "Missing")
)
if (race_sensitivity_only) {
  model_covariates$race_ethnicity_group <- factor(model_covariates$race_ethnicity_group,
    levels=c("White", "Black", "Asian", "Hispanic/Latino", "Other recorded categories", "Unknown/declined/missing"))
  if (anyNA(model_covariates$race_ethnicity_group)) stop("Invalid race/ethnicity group")
}

attach_model_covariates <- function(metrics) {
  join_columns <- c("pair_id", "cohort", "subject_id", "hadm_id")
  merged <- merge(
    metrics,
    model_covariates,
    by = join_columns,
    all.x = TRUE,
    all.y = FALSE,
    sort = FALSE
  )
  required_covariates <- c(
    "age_at_admission_per_10y",
    "elixhauser_score_per_5pt",
    "log1p_prior_all_admissions",
    "language_group"
  )
  if (race_sensitivity_only) required_covariates <- c(required_covariates, "race_ethnicity_group")
  if (any(!complete.cases(merged[, required_covariates, drop = FALSE]))) {
    stop("Some metric rows could not be linked to complete model covariates.")
  }
  merged
}

extract_variance <- function(vc_table, group_name) {
  value <- vc_table$vcov[vc_table$grp == group_name]
  if (length(value) == 0) return(NA_real_)
  as.numeric(value[[1]])
}

fit_one_mixed_model <- function(data, outcome, analysis_level, section_name = NA_character_) {
  model_data <- data[
    is.finite(data[[outcome]]) & !is.na(data$cohort) &
      !is.na(data$patient_cluster) & !is.na(data$pair_cluster),
    ,
    drop = FALSE
  ]
  formula <- as.formula(
    paste0("`", outcome, "` ~ cohort + (1 | patient_cluster) + (1 | pair_cluster)")
  )

  warning_messages <- character(0)
  fit <- withCallingHandlers(
    lmer(
      formula,
      data = model_data,
      REML = TRUE,
      control = lmerControl(
        optimizer = "bobyqa",
        optCtrl = list(maxfun = 200000)
      )
    ),
    warning = function(w) {
      warning_messages <<- c(warning_messages, conditionMessage(w))
      invokeRestart("muffleWarning")
    }
  )

  coefficient_table <- coef(summary(fit))
  term <- "cohortMHC1_psychotic"
  if (!term %in% rownames(coefficient_table)) {
    stop(paste("Missing cohort coefficient for", outcome))
  }
  estimate <- coefficient_table[term, "Estimate"]
  standard_error <- coefficient_table[term, "Std. Error"]
  df <- coefficient_table[term, "df"]
  statistic <- coefficient_table[term, "t value"]
  p_value <- coefficient_table[term, "Pr(>|t|)"]
  critical_value <- qt(0.975, df = df)

  variance_table <- as.data.frame(VarCorr(fit))
  patient_variance <- extract_variance(variance_table, "patient_cluster")
  pair_variance <- extract_variance(variance_table, "pair_cluster")
  residual_variance <- extract_variance(variance_table, "Residual")
  total_variance <- patient_variance + pair_variance + residual_variance

  optimizer_messages <- fit@optinfo$conv$lme4$messages
  optimizer_message <- if (is.null(optimizer_messages)) {
    ""
  } else {
    paste(optimizer_messages, collapse = " | ")
  }

  data.frame(
    analysis_level = analysis_level,
    section_name = section_name,
    outcome = outcome,
    model = "cohort + patient_random_intercept + matched_pair_random_intercept",
    estimation_method = "REML",
    degrees_of_freedom_method = "Satterthwaite",
    n_admissions = nrow(model_data),
    n_patients = length(unique(model_data$patient_cluster)),
    n_matched_pairs = length(unique(model_data$pair_cluster)),
    mhc1_coefficient = estimate,
    standard_error = standard_error,
    degrees_of_freedom = df,
    t_statistic = statistic,
    p_value = p_value,
    ci_low = estimate - critical_value * standard_error,
    ci_high = estimate + critical_value * standard_error,
    patient_variance = patient_variance,
    pair_variance = pair_variance,
    residual_variance = residual_variance,
    patient_variance_fraction = patient_variance / total_variance,
    pair_variance_fraction = pair_variance / total_variance,
    residual_variance_fraction = residual_variance / total_variance,
    singular_fit = isSingular(fit, tol = 1e-4),
    optimizer_message = optimizer_message,
    warnings = paste(unique(warning_messages), collapse = " | "),
    stringsAsFactors = FALSE
  )
}

adjusted_model_definitions <- list(
  M1_age_elixhauser = c(
    "cohort",
    "age_at_admission_per_10y",
    "elixhauser_score_per_5pt"
  ),
  M2_age_elixhauser_prior_utilization = c(
    "cohort",
    "age_at_admission_per_10y",
    "elixhauser_score_per_5pt",
    "log1p_prior_all_admissions"
  ),
  M3_age_elixhauser_prior_utilization_language = c(
    "cohort",
    "age_at_admission_per_10y",
    "elixhauser_score_per_5pt",
    "log1p_prior_all_admissions",
    "language_group"
  )
)

fit_one_adjusted_mixed_model <- function(
    data,
    outcome,
    analysis_level,
    model_name,
    fixed_effects,
    section_name = NA_character_) {
  required_columns <- c(
    outcome,
    fixed_effects,
    "patient_cluster",
    "pair_cluster"
  )
  model_data <- data[complete.cases(data[, required_columns, drop = FALSE]), , drop = FALSE]
  model_data <- model_data[is.finite(model_data[[outcome]]), , drop = FALSE]
  formula <- as.formula(paste0(
    "`", outcome, "` ~ ",
    paste(fixed_effects, collapse = " + "),
    " + (1 | patient_cluster) + (1 | pair_cluster)"
  ))

  warning_messages <- character(0)
  fit <- withCallingHandlers(
    lmer(
      formula,
      data = model_data,
      REML = TRUE,
      control = lmerControl(
        optimizer = "bobyqa",
        optCtrl = list(maxfun = 200000)
      )
    ),
    warning = function(w) {
      warning_messages <<- c(warning_messages, conditionMessage(w))
      invokeRestart("muffleWarning")
    }
  )

  coefficient_table <- coef(summary(fit))
  coefficient_rows <- lapply(
    rownames(coefficient_table),
    function(term) {
      term_df <- coefficient_table[term, "df"]
      term_se <- coefficient_table[term, "Std. Error"]
      critical_value <- qt(0.975, df = term_df)
      data.frame(
        analysis_level = analysis_level,
        section_name = section_name,
        outcome = outcome,
        model = model_name,
        term = term,
        estimate = coefficient_table[term, "Estimate"],
        standard_error = term_se,
        degrees_of_freedom = term_df,
        t_statistic = coefficient_table[term, "t value"],
        p_value = coefficient_table[term, "Pr(>|t|)"],
        ci_low = coefficient_table[term, "Estimate"] - critical_value * term_se,
        ci_high = coefficient_table[term, "Estimate"] + critical_value * term_se,
        stringsAsFactors = FALSE
      )
    }
  )
  coefficients <- do.call(rbind, coefficient_rows)

  variance_table <- as.data.frame(VarCorr(fit))
  patient_variance <- extract_variance(variance_table, "patient_cluster")
  pair_variance <- extract_variance(variance_table, "pair_cluster")
  residual_variance <- extract_variance(variance_table, "Residual")
  total_variance <- patient_variance + pair_variance + residual_variance
  optimizer_messages <- fit@optinfo$conv$lme4$messages
  optimizer_message <- if (is.null(optimizer_messages)) {
    ""
  } else {
    paste(optimizer_messages, collapse = " | ")
  }

  cohort_row <- coefficients[coefficients$term == "cohortMHC1_psychotic", , drop = FALSE]
  if (nrow(cohort_row) != 1) stop(paste("Missing cohort coefficient for", outcome))
  summary <- data.frame(
    analysis_level = analysis_level,
    section_name = section_name,
    outcome = outcome,
    model = model_name,
    fixed_effects = paste(fixed_effects, collapse = " + "),
    estimation_method = "REML",
    degrees_of_freedom_method = "Satterthwaite",
    n_admissions = nrow(model_data),
    n_patients = length(unique(model_data$patient_cluster)),
    n_matched_pairs = length(unique(model_data$pair_cluster)),
    mhc1_coefficient = cohort_row$estimate,
    standard_error = cohort_row$standard_error,
    degrees_of_freedom = cohort_row$degrees_of_freedom,
    t_statistic = cohort_row$t_statistic,
    p_value = cohort_row$p_value,
    ci_low = cohort_row$ci_low,
    ci_high = cohort_row$ci_high,
    patient_variance = patient_variance,
    pair_variance = pair_variance,
    residual_variance = residual_variance,
    patient_variance_fraction = patient_variance / total_variance,
    pair_variance_fraction = pair_variance / total_variance,
    residual_variance_fraction = residual_variance / total_variance,
    singular_fit = isSingular(fit, tol = 1e-4),
    optimizer_message = optimizer_message,
    warnings = paste(unique(warning_messages), collapse = " | "),
    stringsAsFactors = FALSE
  )
  list(summary = summary, coefficients = coefficients)
}

analyze_adjusted_models <- function(
    data,
    outcomes,
    analysis_level,
    section_name = NA_character_) {
  fits <- list()
  index <- 1
  for (model_name in names(adjusted_model_definitions)) {
    for (outcome in outcomes) {
      message(
        "Fitting adjusted model ", model_name, " / ", analysis_level,
        " / ", section_name, " / ", outcome
      )
      fits[[index]] <- fit_one_adjusted_mixed_model(
        data,
        outcome,
        analysis_level,
        model_name,
        adjusted_model_definitions[[model_name]],
        section_name
      )
      index <- index + 1
    }
  }
  list(
    summary = do.call(rbind, lapply(fits, `[[`, "summary")),
    coefficients = do.call(rbind, lapply(fits, `[[`, "coefficients"))
  )
}

paired_tests_for_outcome <- function(data, outcome, analysis_level, section_name = NA_character_) {
  selected <- data[, c("pair_id", "cohort", outcome), drop = FALSE]
  names(selected)[3] <- "value"
  mhc0 <- selected[selected$cohort == "MHC0", c("pair_id", "value")]
  mhc1 <- selected[selected$cohort == "MHC1_psychotic", c("pair_id", "value")]
  names(mhc0)[2] <- "mhc0_value"
  names(mhc1)[2] <- "mhc1_value"
  paired <- merge(mhc0, mhc1, by = "pair_id", all = FALSE)
  paired <- paired[is.finite(paired$mhc0_value) & is.finite(paired$mhc1_value), ]
  difference <- paired$mhc1_value - paired$mhc0_value

  t_result <- t.test(paired$mhc1_value, paired$mhc0_value, paired = TRUE)
  wilcoxon_warning <- ""
  wilcoxon_result <- withCallingHandlers(
    wilcox.test(
      paired$mhc1_value,
      paired$mhc0_value,
      paired = TRUE,
      exact = FALSE,
      correct = TRUE
    ),
    warning = function(w) {
      wilcoxon_warning <<- conditionMessage(w)
      invokeRestart("muffleWarning")
    }
  )

  difference_sd <- sd(difference)
  cohen_dz <- if (is.na(difference_sd) || difference_sd == 0) {
    NA_real_
  } else {
    mean(difference) / difference_sd
  }

  data.frame(
    analysis_level = analysis_level,
    section_name = section_name,
    outcome = outcome,
    n_complete_pairs = nrow(paired),
    mhc0_mean = mean(paired$mhc0_value),
    mhc0_sd = sd(paired$mhc0_value),
    mhc0_median = median(paired$mhc0_value),
    mhc1_mean = mean(paired$mhc1_value),
    mhc1_sd = sd(paired$mhc1_value),
    mhc1_median = median(paired$mhc1_value),
    mean_paired_difference_mhc1_minus_mhc0 = mean(difference),
    median_paired_difference_mhc1_minus_mhc0 = median(difference),
    paired_difference_sd = difference_sd,
    paired_cohen_dz = cohen_dz,
    paired_t_statistic = unname(t_result$statistic),
    paired_t_degrees_of_freedom = unname(t_result$parameter),
    paired_t_p_value = t_result$p.value,
    paired_t_ci_low = t_result$conf.int[[1]],
    paired_t_ci_high = t_result$conf.int[[2]],
    wilcoxon_statistic = unname(wilcoxon_result$statistic),
    wilcoxon_p_value = wilcoxon_result$p.value,
    wilcoxon_warning = wilcoxon_warning,
    stringsAsFactors = FALSE
  )
}

analyze_level <- function(data, outcomes, analysis_level, section_name = NA_character_) {
  mixed_rows <- lapply(
    outcomes,
    function(outcome) {
      message("Fitting ", analysis_level, " / ", section_name, " / ", outcome)
      fit_one_mixed_model(data, outcome, analysis_level, section_name)
    }
  )
  paired_rows <- lapply(
    outcomes,
    function(outcome) paired_tests_for_outcome(
      data,
      outcome,
      analysis_level,
      section_name
    )
  )
  list(
    mixed = do.call(rbind, mixed_rows),
    paired = do.call(rbind, paired_rows)
  )
}

if (race_sensitivity_only && Sys.getenv("LANGUAGE_COMPLEXITY_FUNCTIONS_ONLY") != "1") {
  m3 <- adjusted_model_definitions$M3_age_elixhauser_prior_utilization_language
  adjusted_model_definitions <- list(
    M3_age_elixhauser_prior_utilization_language=m3,
    M4_M3_plus_race_ethnicity=c(m3, "race_ethnicity_group"))
  full <- attach_model_covariates(attach_pair_ids(read.csv(full_note_path,check.names=FALSE)))
  sections <- attach_model_covariates(attach_pair_ids(read.csv(section_path,check.names=FALSE)))
  stopifnot(nrow(full) == 2*nrow(pairs))
  results <- list(analyze_adjusted_models(full, full_note_outcomes, "full_note"))
  for (section in sort(unique(sections$section_name))) {
    results[[length(results)+1]] <- analyze_adjusted_models(
      sections[sections$section_name == section,], section_outcomes, "prose_section", section)
  }
  summaries <- do.call(rbind,lapply(results, `[[`, "summary"))
  coefficients <- do.call(rbind,lapply(results, `[[`, "coefficients"))
  family <- interaction(summaries$analysis_level, ifelse(is.na(summaries$section_name),"full_note",summaries$section_name), summaries$model, drop=TRUE)
  summaries$FDR <- ave(summaries$p_value,family,FUN=function(x) p.adjust(x,"BH"))
  m3rows <- summaries[summaries$model == names(adjusted_model_definitions)[1],]
  m4rows <- summaries[summaries$model == names(adjusted_model_definitions)[2],]
  # NA section keys become an explicit label for an exact one-to-one comparison.
  m3rows$section_name[is.na(m3rows$section_name)] <- "full_note"
  m4rows$section_name[is.na(m4rows$section_name)] <- "full_note"
  keys <- c("analysis_level", "section_name", "outcome")
  keep <- c(keys,"n_admissions","mhc1_coefficient","ci_low","ci_high","FDR","singular_fit","optimizer_message","warnings")
  comparison <- merge(m3rows[,keep],m4rows[,keep],by=keys,suffixes=c("_M3","_M4"))
  stopifnot(nrow(comparison) == nrow(m3rows), all(comparison$n_admissions_M3 == comparison$n_admissions_M4))
  comparison$coefficient_change <- comparison$mhc1_coefficient_M4 - comparison$mhc1_coefficient_M3
  write.csv(summaries,file.path(output_dir,"race_sensitivity_cohort_effects.csv"),row.names=FALSE)
  write.csv(coefficients,file.path(output_dir,"race_sensitivity_coefficients.csv"),row.names=FALSE)
  write.csv(comparison,file.path(output_dir,"M3_M4_cohort_effect_comparison.csv"),row.names=FALSE)
  writeLines(c(
    "Exploratory race/ethnicity sensitivity: M4 = M3 + six-category recorded race/ethnicity.",
    "M3: cohort + age (per 10y) + Elixhauser (per 5pt) + log1p(prior admissions) + language.",
    "Both models retain patient and matched-pair random intercepts; REML, Satterthwaite df, t-based 95% intervals.",
    "M3 rerun on identical admissions to M4. Existing M0-M3 outputs remain unchanged.",
    "White is race reference; Black, Asian, Hispanic/Latino, other recorded categories, unknown/declined/missing.",
    "Other combines less common recorded categories; unknown is retained as a category, not imputed identity.",
    "Mapping reuses cohort-characterization rules; raw MIMIC race field combines racial/ethnic designations.",
    "BH FDR across all original outcomes within each model and section (11 full-note, 9 per prose section).",
    "This is covariate adjustment, not a test of racial discrimination, effect modification, or a causal psychosis effect."
  ),file.path(output_dir,"race_sensitivity_methods.txt"))
  capture.output(sessionInfo(),file=file.path(output_dir,"R_session_info.txt"))
  cat("Saved race sensitivity results to",output_dir,"\n")
} else if (Sys.getenv("LANGUAGE_COMPLEXITY_FUNCTIONS_ONLY") != "1") {
full_note_metrics <- attach_model_covariates(
  attach_pair_ids(read.csv(full_note_path, check.names = FALSE))
)
if (nrow(full_note_metrics) != 2 * nrow(pairs)) {
  stop("Full-note metrics do not contain exactly two rows per matched pair.")
}
full_note_results <- analyze_level(
  full_note_metrics,
  full_note_outcomes,
  "full_note"
)
full_note_adjusted <- analyze_adjusted_models(
  full_note_metrics,
  full_note_outcomes,
  "full_note"
)

section_metrics <- attach_model_covariates(
  attach_pair_ids(read.csv(section_path, check.names = FALSE))
)
section_result_list <- lapply(
  sort(unique(section_metrics$section_name)),
  function(section_name) {
    section_data <- section_metrics[section_metrics$section_name == section_name, ]
    analyze_level(section_data, section_outcomes, "prose_section", section_name)
  }
)
section_mixed <- do.call(rbind, lapply(section_result_list, `[[`, "mixed"))
section_paired <- do.call(rbind, lapply(section_result_list, `[[`, "paired"))
section_adjusted_result_list <- lapply(
  sort(unique(section_metrics$section_name)),
  function(section_name) {
    section_data <- section_metrics[section_metrics$section_name == section_name, ]
    analyze_adjusted_models(
      section_data,
      section_outcomes,
      "prose_section",
      section_name
    )
  }
)
section_adjusted_summary <- do.call(
  rbind,
  lapply(section_adjusted_result_list, `[[`, "summary")
)
section_adjusted_coefficients <- do.call(
  rbind,
  lapply(section_adjusted_result_list, `[[`, "coefficients")
)

add_fdr <- function(data, group_columns, p_column, output_column) {
  interaction_group <- interaction(data[, group_columns, drop = FALSE], drop = TRUE)
  data[[output_column]] <- ave(
    data[[p_column]],
    interaction_group,
    FUN = function(values) p.adjust(values, method = "BH")
  )
  data
}

full_note_results$mixed$p_value_fdr_bh <- p.adjust(
  full_note_results$mixed$p_value,
  method = "BH"
)
full_note_results$paired$paired_t_p_value_fdr_bh <- p.adjust(
  full_note_results$paired$paired_t_p_value,
  method = "BH"
)
full_note_results$paired$wilcoxon_p_value_fdr_bh <- p.adjust(
  full_note_results$paired$wilcoxon_p_value,
  method = "BH"
)

section_mixed <- add_fdr(
  section_mixed,
  c("section_name"),
  "p_value",
  "p_value_fdr_bh"
)
section_paired <- add_fdr(
  section_paired,
  c("section_name"),
  "paired_t_p_value",
  "paired_t_p_value_fdr_bh"
)
full_note_adjusted$summary <- add_fdr(
  full_note_adjusted$summary,
  c("model"),
  "p_value",
  "p_value_fdr_bh"
)
section_adjusted_summary <- add_fdr(
  section_adjusted_summary,
  c("section_name", "model"),
  "p_value",
  "p_value_fdr_bh"
)
section_paired <- add_fdr(
  section_paired,
  c("section_name"),
  "wilcoxon_p_value",
  "wilcoxon_p_value_fdr_bh"
)

write.csv(
  full_note_results$mixed,
  file.path(output_dir, "full_note_simple_mixed_models.csv"),
  row.names = FALSE
)
write.csv(
  full_note_results$paired,
  file.path(output_dir, "full_note_paired_tests.csv"),
  row.names = FALSE
)
write.csv(
  section_mixed,
  file.path(output_dir, "prose_section_simple_mixed_models.csv"),
  row.names = FALSE
)
write.csv(
  section_paired,
  file.path(output_dir, "prose_section_paired_tests.csv"),
  row.names = FALSE
)
write.csv(
  full_note_adjusted$summary,
  file.path(output_dir, "full_note_adjusted_mixed_models.csv"),
  row.names = FALSE
)
write.csv(
  full_note_adjusted$coefficients,
  file.path(output_dir, "full_note_adjusted_mixed_model_coefficients.csv"),
  row.names = FALSE
)
write.csv(
  section_adjusted_summary,
  file.path(output_dir, "prose_section_adjusted_mixed_models.csv"),
  row.names = FALSE
)
write.csv(
  section_adjusted_coefficients,
  file.path(output_dir, "prose_section_adjusted_mixed_model_coefficients.csv"),
  row.names = FALSE
)

cat("\nSaved language-complexity inference outputs to:\n", output_dir, "\n", sep = "")
cat("\nFull-note mixed-model cohort effects:\n")
print(
  full_note_results$mixed[, c(
    "outcome", "mhc1_coefficient", "ci_low", "ci_high",
    "p_value_fdr_bh", "patient_variance_fraction",
    "pair_variance_fraction", "singular_fit"
  )],
  row.names = FALSE
)
cat("\nFull-note paired tests:\n")
print(
  full_note_results$paired[, c(
    "outcome", "n_complete_pairs",
    "mean_paired_difference_mhc1_minus_mhc0", "paired_cohen_dz",
    "paired_t_p_value_fdr_bh", "wilcoxon_p_value_fdr_bh"
  )],
  row.names = FALSE
)
cat("\nFull-note adjusted mixed-model cohort effects:\n")
print(
  full_note_adjusted$summary[, c(
    "model", "outcome", "mhc1_coefficient", "ci_low", "ci_high",
    "p_value_fdr_bh", "singular_fit"
  )],
  row.names = FALSE
)
}
