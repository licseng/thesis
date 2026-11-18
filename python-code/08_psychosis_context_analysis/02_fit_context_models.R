#!/usr/bin/env Rscript
# Exploratory three-group comparisons; M0, M1 and M2 use identical endpoint samples.
# Readability uses OLS, mortality uses logistic regression, and events/durations use PPML.
# Exploratory cohort contrasts use BH adjustment within outcome family and stage.
file_arg <- grep("^--file=", commandArgs(FALSE), value=TRUE)
here <- dirname(normalizePath(sub("^--file=", "", file_arg)))
python <- dirname(here)
.libPaths(c(file.path(python,
 "03_discharge_note_text_analysis/01_language_complexity/.r-library"), .libPaths()))
output <- file.path(here, "analysis_output")
clinical <- "--clinical-events" %in% commandArgs(TRUE)
if(clinical) output <- file.path(output,"clinical_activity")
los_sensitivity <- "--mortality-los-sensitivity" %in% commandArgs(TRUE)
stopifnot(!(clinical && los_sensitivity))
input_dir <- output
if(los_sensitivity) output <- file.path(output,"mortality_los_sensitivity")

outcomes <- list(
  readability=c("discharge_instructions_flesch_reading_ease", "whole_note_flesch_reading_ease",
                "discharge_instructions_automated_readability_index", "whole_note_automated_readability_index"),
  duration=c("hospital_los_days", "ed_los_hours"),
  mortality=c("in_hospital_mortality", "post_discharge_death_within_1y"),
  clinical_activity=c("laboratory_measurements", "laboratory_specimens",
                      "microbiology_records", "provider_order_activity"))
families <- if(clinical) "clinical_activity" else if(los_sensitivity) "mortality" else c("readability", "duration", "mortality")

save_context_contrasts <- function(results) {
 contrasts <- c("history_only_vs_MHC0", "current_vs_MHC0", "current_vs_history_only")
 stages <- if(los_sensitivity) c("M2", "M2_plus_LOS") else c("M0", "M1", "M2")
 stopifnot(setequal(results$outcome, unlist(outcomes[families], use.names=FALSE)),
           setequal(results$stage, stages), setequal(results$contrast, contrasts),
           !anyDuplicated(results[,c("outcome", "stage", "contrast")]))
 results$p_value <- results$p_raw_exploratory
 results$FDR <- NA_real_
 results$analysis_family <- NA_character_
 for(family in families) for(stage in stages) {
  ix <- results$outcome %in% outcomes[[family]] & results$stage==stage
  n_tests <- length(outcomes[[family]])*length(contrasts)
  stopifnot(sum(ix)==n_tests)
  results$FDR[ix] <- p.adjust(results$p_value[ix], method="BH", n=n_tests)
  results$analysis_family[ix] <- family
 }
 write.csv(results,file.path(output,"group_contrasts.csv"),row.names=FALSE)
 writeLines(c("Exploratory cohort contrasts: BH adjustment separately within each model stage and outcome family.",
   "Readability: four outcomes x three contrasts. Duration and mortality: two outcomes x three contrasts each.",
   "Clinical activity: four outcomes x three contrasts.",
   "Unavailable p-values count toward the planned family size and remain missing.",
   "p_value and p_raw_exploratory are the same raw p-value. FDR is the BH-adjusted p-value. Confidence intervals are pointwise."),
   file.path(output,"multiplicity_methods.txt"))
}

if("--adjust-p-values-only" %in% commandArgs(TRUE)) {
 save_context_contrasts(read.csv(file.path(output,"group_contrasts.csv")))
 quit(save="no")
}
suppressPackageStartupMessages({library(sandwich)})
d_all <- read.csv(file.path(input_dir, "model_inputs.csv"))
if(los_sensitivity) {
 keys <- c("pair_id","cohort","subject_id","hadm_id","context_group")
 stay <- d_all[d_all$outcome=="hospital_los_days",c(keys,"value")]
 stopifnot(!anyDuplicated(stay[keys]),all(is.finite(stay$value)),all(stay$value>0))
 stay$log1p_current_los_days <- log1p(stay$value)
 mortality_inputs <- d_all[d_all$outcome %in% outcomes$mortality,]
 d_all <- merge(mortality_inputs,stay[c(keys,"log1p_current_los_days")],by=keys,all.x=TRUE)
 stopifnot(nrow(d_all)==nrow(mortality_inputs))
 d_all <- do.call(rbind,lapply(split(d_all,d_all$outcome),function(d) {
  keep <- tapply(is.finite(d$log1p_current_los_days),d$pair_id,sum)
  d[d$pair_id %in% names(keep)[keep==2],]
 }))
 dir.create(output,recursive=TRUE,showWarnings=FALSE)
}
if(clinical) {
 d_all$outcome <- d_all$measure
 d_all$value <- d_all$n_events
 d_all$period <- factor(d_all$interval,
  levels=c("0–24h","24–48h","48–72h","days 4–7","after day 7"))
 stopifnot(all(is.finite(d_all$exposure_days)),all(d_all$exposure_days>0),
  all(d_all$value>=0),all(d_all$value==round(d_all$value)),
  !anyDuplicated(d_all[,c("hadm_id","measure","interval")]))
}
d_all$context_group <- factor(d_all$context_group,
 levels=c("MHC0","history_only","current_with_or_without_history"))
# Patient IDs, not subgroup labels, identify the same person across subgroups.
d_all$patient <- factor(d_all$subject_id)
d_all$matched_pair <- factor(d_all$pair_id)
adjustments <- c("age_at_admission_per_10y", "elixhauser_score_per_5pt",
 "log1p_prior_all_admissions")
model_adjustments <- list(M0=character(), M1=adjustments[1:2], M2=adjustments)
if(los_sensitivity) model_adjustments <- list(M2=adjustments,
 M2_plus_LOS=c(adjustments,"log1p_current_los_days"))
results <- list(); diagnostics <- list(); coefficients <- list(); calibration <- list()
for (outcome in unique(d_all$outcome)) for (stage in names(model_adjustments)) {
 d <- droplevels(d_all[d_all$outcome==outcome,])
 admissions <- unique(d[,c("hadm_id","matched_pair","context_group")])
 stopifnot(!anyNA(d), all(table(admissions$matched_pair)==2),
  all(tapply(admissions$context_group=="MHC0",admissions$matched_pair,sum)==1))
 fixed <- c("context_group", model_adjustments[[stage]],
  if(clinical)c("period","offset(log(exposure_days))"))
 stopifnot(outcome %in% unlist(outcomes[families], use.names=FALSE))
 linear <- outcome %in% outcomes$readability
 mortality <- outcome %in% outcomes$mortality
 if(mortality) stopifnot(all(d$value %in% c(0,1)))
 form <- reformulate(fixed, response="value")
 warnings <- character()
 fit <- withCallingHandlers(
  if(linear) lm(form, data=d)
  else glm(form, data=d, family=if(mortality) binomial() else quasipoisson(link="log"),
   control=glm.control(maxit=100,epsilon=1e-10)),
  warning=function(w){warnings <<- c(warnings,conditionMessage(w));invokeRestart("muffleWarning")})
 b <- coef(fit)
 v <- vcovCL(fit,
  cluster=d[,c("patient","matched_pair")], type="HC1",
  cadjust=TRUE, multi0=FALSE, fix=FALSE)
 {
  # Independently verify the two-way sandwich including observed intersections.
  x <- model.matrix(fit)
  weights <- if(linear) rep(1,nrow(d)) else as.numeric(fit$weights)
  bread <- solve(crossprod(x,x*weights))
  scores <- x*as.numeric(fit$residuals*weights)
  component <- function(g) {
   s <- rowsum(scores,g,reorder=FALSE); G <- nrow(s)
   crossprod(s)*G/(G-1)*(nrow(x)-1)/(nrow(x)-ncol(x))
  }
  intersections <- paste(d$patient,d$matched_pair,sep=":")
  manual <- bread %*% (component(d$patient)+component(d$matched_pair)-
   component(intersections)) %*% bread
  stopifnot(isTRUE(all.equal(unname(v),unname(manual),tolerance=1e-7)))
  if(stage=="M0" && !clinical) {
   means <- tapply(d$value,d$context_group,mean)
   if(mortality) means <- means/(1-means)
   observed <- if(linear) means[2:3]-means[1] else means[2:3]/means[1]
   estimated <- if(linear) b[2:3] else exp(b[2:3])
   stopifnot(isTRUE(all.equal(as.numeric(estimated),as.numeric(observed),tolerance=1e-7)))
  }
 }
 eigenvalues <- eigen(v,symmetric=TRUE,only.values=TRUE)$values
 valid <- all(is.finite(b)) && all(is.finite(v)) &&
  all(diag(v)>0) && min(eigenvalues)>=-1e-8*max(abs(eigenvalues))
 messages <- character()
 converged <- if(linear) fit$rank==ncol(x) else fit$converged
 valid <- valid && fit$rank==ncol(x)
 valid <- valid && isTRUE(converged)
 singular <- NA
 warning_flag <- !valid || length(warnings)>0 || isTRUE(singular)
 key <- paste(outcome,stage,sep="__")
 variance <- function(group) NA_real_
 diagnostics[[key]] <- data.frame(outcome=outcome,stage=stage,
  n_admissions=nrow(admissions),n_observations=nrow(d),n_pairs=nlevels(d$matched_pair),
  n_patients=nlevels(d$patient),converged=converged,
  singular=singular,numeric_valid=valid,warning_flag=warning_flag,
  patient_variance=variance("patient"),pair_variance=variance("matched_pair"),
  residual_variance=variance("Residual"),covariance_min_eigenvalue=min(eigenvalues),
  warnings=paste(c(warnings,messages),collapse=" | "),
  n_positive_admissions=if(mortality) sum(d$value) else NA_integer_,
  framework=paste(if(linear) "OLS" else if(mortality) "Logistic regression" else "PPML",
                  "two-way patient/pair HC1 SE", sep="; "))
 terms <- c("context_grouphistory_only",
  "context_groupcurrent_with_or_without_history")
 stopifnot(all(terms %in% names(b)))
 contrasts <- list(history_only_vs_MHC0=c(1,0),
  current_vs_MHC0=c(0,1),current_vs_history_only=c(-1,1))
 for (label in names(contrasts)) {
  L <- setNames(rep(0,length(b)),names(b))
  L[terms] <- contrasts[[label]]
  beta <- sum(L*b)
  se <- sqrt(as.numeric(t(L)%*%v%*%L))
  df <- Inf
  cutoff <- qnorm(.975)
  p <- if(valid)2*pt(-abs(beta/se),df=df) else NA_real_
  lo <- beta-cutoff*se; hi <- beta+cutoff*se
  results[[paste(key,label)]] <- data.frame(outcome=outcome,stage=stage,
   contrast=label,beta=beta,se=se,df=df,
   effect=if(linear)beta else exp(beta),
   ci_low=if(linear)lo else exp(lo),ci_high=if(linear)hi else exp(hi),
   p_raw_exploratory=p,warning_flag=warning_flag,
   effect_type=if(clinical)"event rate ratio" else if(mortality)"mortality odds ratio" else
    if(linear && grepl("automated_readability_index$",outcome))"Automated Readability Index point difference" else
    if(linear)"Flesch reading ease point difference" else
    if(outcome=="ed_los_hours")"mean ED stay ratio" else "mean LOS ratio")
 }
 coefficients[[key]] <- data.frame(outcome=outcome,stage=stage,
  term=names(b),beta=unname(b),se=sqrt(diag(v)),warning_flag=warning_flag)
 if(clinical) {
  d$predicted_events <- fitted(fit)
  cal <- aggregate(cbind(n_events,predicted_events,exposure_days) ~ context_group+period,d,sum)
  cal$observed_rate <- cal$n_events/cal$exposure_days
  cal$predicted_rate <- cal$predicted_events/cal$exposure_days
  cal$outcome <- outcome; cal$stage <- stage
  calibration[[key]] <- cal
 }
 cat(outcome,stage,":",nrow(admissions),"admissions; warning flag",warning_flag,"\n")
}
save_context_contrasts(do.call(rbind,results))
write.csv(do.call(rbind,diagnostics),file.path(output,"model_diagnostics.csv"),row.names=FALSE)
write.csv(do.call(rbind,coefficients),file.path(output,"model_coefficients.csv"),row.names=FALSE)
if(clinical)write.csv(do.call(rbind,calibration),
 file.path(output,"mean_calibration_by_group_period.csv"),row.names=FALSE)
capture.output(sessionInfo(),file=file.path(output,"R_session_info.txt"))

if(los_sensitivity) {
 effects <- read.csv(file.path(output,"group_contrasts.csv"))
 diagnostics <- read.csv(file.path(output,"model_diagnostics.csv"))
 stopifnot(nrow(effects)==12, all(diagnostics$numeric_valid), !any(diagnostics$warning_flag),
           all(tapply(diagnostics$n_pairs, diagnostics$outcome, function(n) length(unique(n)))==1))
 keys <- c("outcome","contrast")
 columns <- c(keys,"effect","ci_low","ci_high","p_value","FDR")
 comparison <- merge(effects[effects$stage=="M2",columns],
                     effects[effects$stage=="M2_plus_LOS",columns],
                     by=keys,suffixes=c("_M2","_plus_LOS"))
 old <- read.csv(file.path(input_dir,"group_contrasts.csv"))
 old <- old[old$stage=="M2" & old$outcome %in% outcomes$mortality,columns]
 names(old)[-(1:2)] <- paste0(names(old)[-(1:2)],"_original_M2")
 comparison <- merge(comparison,old,by=keys)
 comparison$crossed_05 <- (comparison$FDR_M2<.05)!=(comparison$FDR_plus_LOS<.05)
 comparison$crossed_05_vs_original <- (comparison$FDR_original_M2<.05)!=(comparison$FDR_plus_LOS<.05)
 write.csv(comparison,file.path(output,"comparison.csv"),row.names=FALSE)
 writeLines(c("Exploratory context mortality models: M2 and M2 plus log(1 + current hospital LOS in days).",
  "Logistic regression, two-way patient/pair HC1 covariance, identical complete pairs per endpoint.",
  "BH across two mortality endpoints and all three contrasts per model stage (six tests each)."),
  file.path(output,"multiplicity_methods.txt"))
 print(comparison[,c(keys,"effect_M2","effect_plus_LOS","ci_low_plus_LOS","ci_high_plus_LOS",
                     "FDR_M2","FDR_plus_LOS","crossed_05","crossed_05_vs_original")],row.names=FALSE)
 print(diagnostics[,c("outcome","stage","n_pairs","n_positive_admissions")],row.names=FALSE)
}
