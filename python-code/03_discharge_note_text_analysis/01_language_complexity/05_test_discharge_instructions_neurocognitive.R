#!/usr/bin/env Rscript
# Exploratory sensitivity: prior/current secondary neurocognitive category.
# Scripts and outputs live together in the language-complexity subfolder.
# Each specification retains patient and matched-pair random intercepts.
args <- commandArgs(FALSE)
script_dir <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", args, value=TRUE))))
Sys.setenv(LANGUAGE_COMPLEXITY_FUNCTIONS_ONLY="1")
source(file.path(script_dir, "03_test_language_complexity.R"))
Sys.unsetenv("LANGUAGE_COMPLEXITY_FUNCTIONS_ONLY")
output_dir <- file.path(script_dir, "analysis_output_neurocognitive_sensitivity")
flags <- read.csv(file.path(output_dir, "neurocognitive_model_covariates.csv"))
data <- attach_model_covariates(attach_pair_ids(read.csv(section_path)))
data <- data[data$section_name == "discharge_instructions", ]
data <- merge(data, flags[, c("cohort", "subject_id", "hadm_id", "neurocognitive")],
              by=c("cohort", "subject_id", "hadm_id"), all.x=TRUE, sort=FALSE)
stopifnot(!anyNA(data$neurocognitive), !anyDuplicated(data[, c("cohort", "hadm_id")]))
# Drop entire pairs whose MHC1 admission has neurocognitive context, preserving
# original matching for the exclusion sensitivity.
excluded_pairs <- flags$pair_id[flags$cohort == "MHC1_psychotic" & flags$neurocognitive == 1]
fits <- list()
for (base in c("M2_age_elixhauser_prior_utilization", "M3_age_elixhauser_prior_utilization_language")) {
  for (spec in c("baseline", "plus_neurocognitive", "exclude_neurocognitive_pairs")) {
    selected <- if (spec == "exclude_neurocognitive_pairs") data[!data$pair_id %in% excluded_pairs, ] else data
    fixed <- adjusted_model_definitions[[base]]
    if (spec == "plus_neurocognitive") fixed <- c(fixed, "neurocognitive")
    for (outcome in section_outcomes) {
      message(base, " / ", spec, " / ", outcome)
      fits[[length(fits)+1]] <- fit_one_adjusted_mixed_model(
        selected, outcome, "prose_section", paste(base, spec, sep="__"), fixed, "discharge_instructions")
    }
  }
}
summary <- do.call(rbind, lapply(fits, `[[`, "summary"))
coefficients <- do.call(rbind, lapply(fits, `[[`, "coefficients"))
summary$FDR <- ave(summary$p_value, summary$model, FUN=function(p) p.adjust(p, "BH"))
coefficients$FDR <- ave(coefficients$p_value, interaction(coefficients$model, coefficients$term), FUN=function(p) p.adjust(p, "BH"))
write.csv(summary, file.path(output_dir, "discharge_instructions_neurocognitive_models.csv"), row.names=FALSE)
write.csv(coefficients, file.path(output_dir, "discharge_instructions_neurocognitive_coefficients.csv"), row.names=FALSE)
coverage <- aggregate(neurocognitive ~ cohort, data, function(x) c(n=length(x), n_neurocognitive=sum(x)))
write.csv(coverage, file.path(output_dir, "discharge_instructions_neurocognitive_coverage.csv"), row.names=FALSE)
print(summary[, c("model", "outcome", "n_admissions", "mhc1_coefficient", "ci_low", "ci_high", "FDR", "singular_fit")], row.names=FALSE)
