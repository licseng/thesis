#!/usr/bin/env Rscript
# Aggregate MHC1-psychosis vs MHC0 PPML; raw exploratory p-values.
# Invoked by 03_analyze_chief_complaint_subgroup_outcomes.py.
file_arg <- grep("^--file=",commandArgs(FALSE),value=TRUE)
here <- dirname(normalizePath(sub("^--file=","",file_arg)))
python <- dirname(here)
.libPaths(c(file.path(python,
 "03_discharge_note_text_analysis/01_language_complexity/.r-library"),.libPaths()))
suppressPackageStartupMessages(library(sandwich))
output <- file.path(here,"analysis_output_whole_matched_clinical_activity")
data <- read.csv(file.path(output,"model_inputs.csv"))
data$outcome <- data$measure; data$value <- data$n_events
data$period <- factor(data$interval,
 levels=c("0–24h","24–48h","48–72h","days 4–7","after day 7"))
data$context_group <- factor(data$cohort,levels=c("MHC0","MHC1_psychotic"))
data$patient <- factor(data$subject_id); data$matched_pair <- factor(data$pair_id)
data$language_group <- factor(data$language_group,levels=c("English","Non-English","Missing"))
data$race_ethnicity_group <- factor(data$race_ethnicity_group,
 levels=c("White","Black","Asian","Hispanic/Latino","Other recorded categories",
 "Unknown/declined/missing"))
stopifnot(!anyNA(data),all(is.finite(data$exposure_days)),all(data$exposure_days>0),
 all(data$value>=0),all(data$value==round(data$value)),
 !anyDuplicated(data[,c("hadm_id","measure","interval")]))
adjustments <- c("age_at_admission_per_10y","elixhauser_score_per_5pt",
 "log1p_prior_all_admissions")
models <- list(M0=character(),M2=adjustments,M3=c(adjustments,"language_group"),
 M4=c(adjustments,"language_group","race_ethnicity_group"))
effects <- list(); diagnostics <- list(); coefficients <- list(); calibration <- list()
for(outcome in unique(data$outcome)) for(stage in names(models)) {
 d <- droplevels(data[data$outcome==outcome,])
 admissions <- unique(d[,c("hadm_id","matched_pair","context_group")])
 stopifnot(all(table(admissions$matched_pair)==2),
  all(tapply(admissions$context_group=="MHC0",admissions$matched_pair,sum)==1))
 form <- reformulate(c("context_group",models[[stage]],"period",
  "offset(log(exposure_days))"),response="value")
 warnings <- character()
 fit <- withCallingHandlers(glm(form,data=d,family=quasipoisson(link="log"),
  control=glm.control(maxit=100,epsilon=1e-10)),
  warning=function(w){warnings <<- c(warnings,conditionMessage(w));invokeRestart("muffleWarning")})
 b <- coef(fit)
 v <- vcovCL(fit,cluster=d[,c("patient","matched_pair")],type="HC1",
  cadjust=TRUE,multi0=FALSE,fix=FALSE)
 # Independently verify two-way covariance with observed intersections.
 x <- model.matrix(fit)
 bread <- solve(crossprod(x,x*as.numeric(fit$weights)))
 scores <- x*as.numeric(fit$residuals*fit$weights)
 component <- function(g) {
  s <- rowsum(scores,g,reorder=FALSE); G <- nrow(s)
  crossprod(s)*G/(G-1)*(nrow(x)-1)/(nrow(x)-ncol(x))
 }
 manual <- bread %*% (component(d$patient)+component(d$matched_pair)-
  component(paste(d$patient,d$matched_pair,sep=":"))) %*% bread
 stopifnot(isTRUE(all.equal(unname(v),unname(manual),tolerance=1e-7)))
 eigenvalues <- eigen(v,symmetric=TRUE,only.values=TRUE)$values
 valid <- isTRUE(fit$converged) && fit$rank==ncol(x) &&
  all(is.finite(b)) && all(is.finite(v)) && all(diag(v)>0) &&
  min(eigenvalues)>=-1e-8*max(abs(eigenvalues))
 flag <- !valid || length(warnings)>0
 key <- paste(outcome,stage,sep="__")
 diagnostics[[key]] <- data.frame(outcome=outcome,stage=stage,
  n_admissions=nrow(admissions),n_observations=nrow(d),n_pairs=nlevels(d$matched_pair),
  n_patients=nlevels(d$patient),converged=fit$converged,singular=NA,
  numeric_valid=valid,warning_flag=flag,patient_variance=NA_real_,
  pair_variance=NA_real_,residual_variance=NA_real_,
  covariance_min_eigenvalue=min(eigenvalues),warnings=paste(warnings,collapse=" | "),
  framework="PPML; two-way patient/pair HC1 SE")
 term <- "context_groupMHC1_psychotic"
 beta <- unname(b[term]); se <- sqrt(v[term,term])
 effects[[key]] <- data.frame(outcome=outcome,stage=stage,
  contrast="MHC1_psychosis_vs_MHC0",beta=beta,se=se,df=Inf,
  effect=exp(beta),ci_low=exp(beta-qnorm(.975)*se),
  ci_high=exp(beta+qnorm(.975)*se),
  p_raw_exploratory=if(valid)2*pnorm(-abs(beta/se)) else NA_real_,
  warning_flag=flag,effect_type="event rate ratio")
 coefficients[[key]] <- data.frame(outcome=outcome,stage=stage,
  term=names(b),beta=unname(b),se=sqrt(diag(v)),warning_flag=flag)
 d$predicted_events <- fitted(fit)
 cal <- aggregate(cbind(n_events,predicted_events,exposure_days) ~ context_group+period,d,sum)
 cal$observed_rate <- cal$n_events/cal$exposure_days
 cal$predicted_rate <- cal$predicted_events/cal$exposure_days
 cal$outcome <- outcome; cal$stage <- stage; calibration[[key]] <- cal
 cat(outcome,stage,":",nrow(admissions),"admissions; warning flag",flag,"\n")
}
write.csv(do.call(rbind,effects),file.path(output,"group_contrasts.csv"),row.names=FALSE)
write.csv(do.call(rbind,diagnostics),file.path(output,"model_diagnostics.csv"),row.names=FALSE)
write.csv(do.call(rbind,coefficients),file.path(output,"model_coefficients.csv"),row.names=FALSE)
write.csv(do.call(rbind,calibration),file.path(output,"mean_calibration_by_group_period.csv"),row.names=FALSE)
capture.output(sessionInfo(),file=file.path(output,"R_session_info.txt"))
