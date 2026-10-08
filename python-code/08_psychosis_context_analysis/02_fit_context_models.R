#!/usr/bin/env Rscript
# Exploratory three-group comparisons; M0, M1 and M2 use identical endpoint samples.
# No multiplicity adjustment: raw p-values are exploratory, not confirmatory.
file_arg <- grep("^--file=", commandArgs(FALSE), value=TRUE)
here <- dirname(normalizePath(sub("^--file=", "", file_arg)))
python <- dirname(here)
.libPaths(c(file.path(python,
 "03_discharge_note_text_analysis/01_language_complexity/.r-library"), .libPaths()))
suppressPackageStartupMessages({library(lmerTest); library(sandwich)})
output <- file.path(here, "analysis_output")
clinical <- "--clinical-events" %in% commandArgs(TRUE)
if(clinical) output <- file.path(output,"clinical_activity")
d_all <- read.csv(file.path(output, "model_inputs.csv"))
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
results <- list(); diagnostics <- list(); coefficients <- list(); calibration <- list()
for (outcome in unique(d_all$outcome)) for (stage in names(model_adjustments)) {
 d <- droplevels(d_all[d_all$outcome==outcome,])
 admissions <- unique(d[,c("hadm_id","matched_pair","context_group")])
 stopifnot(!anyNA(d), all(table(admissions$matched_pair)==2),
  all(tapply(admissions$context_group=="MHC0",admissions$matched_pair,sum)==1))
 fixed <- c("context_group", model_adjustments[[stage]],
  if(clinical)c("period","offset(log(exposure_days))"))
 stopifnot(outcome %in% c("laboratory_measurements","laboratory_specimens",
  "microbiology_records","provider_order_activity","discharge_instructions_flesch_reading_ease",
  "whole_note_flesch_reading_ease", "hospital_los_days", "ed_los_hours"))
 mixed <- outcome %in% c("discharge_instructions_flesch_reading_ease",
  "whole_note_flesch_reading_ease")
 form <- reformulate(c(fixed, if(mixed)
  c("(1 | patient)","(1 | matched_pair)")), response="value")
 warnings <- character()
 fit <- withCallingHandlers(
  if(mixed) lmer(form, data=d, REML=TRUE,
   control=lmerControl(optimizer="bobyqa",optCtrl=list(maxfun=200000)))
  else glm(form, data=d, family=quasipoisson(link="log"),
   control=glm.control(maxit=100,epsilon=1e-10)),
  warning=function(w){warnings <<- c(warnings,conditionMessage(w));invokeRestart("muffleWarning")})
 b <- if(mixed)fixef(fit) else coef(fit)
 v <- if(mixed)as.matrix(vcov(fit)) else vcovCL(fit,
  cluster=d[,c("patient","matched_pair")], type="HC1",
  cadjust=TRUE, multi0=FALSE, fix=FALSE)
 if(!mixed) {
  # Independently verify the two-way sandwich including observed intersections.
  x <- model.matrix(fit)
  bread <- solve(crossprod(x,x*as.numeric(fit$weights)))
  scores <- x*as.numeric(fit$residuals*fit$weights)
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
   stopifnot(isTRUE(all.equal(as.numeric(exp(b[2:3])),
    as.numeric(means[2:3]/means[1]),tolerance=1e-7)))
  }
 }
 eigenvalues <- eigen(v,symmetric=TRUE,only.values=TRUE)$values
 valid <- all(is.finite(b)) && all(is.finite(v)) &&
  all(diag(v)>0) && min(eigenvalues)>=-1e-8*max(abs(eigenvalues))
 messages <- if(mixed)fit@optinfo$conv$lme4$messages else character()
 converged <- if(mixed)is.null(messages) && fit@optinfo$conv$opt==0 else fit$converged
 valid <- valid && isTRUE(converged)
 singular <- if(mixed)isSingular(fit,tol=1e-4) else NA
 warning_flag <- !valid || length(warnings)>0 || isTRUE(singular)
 key <- paste(outcome,stage,sep="__")
 vc <- if(mixed)as.data.frame(VarCorr(fit)) else NULL
 variance <- function(group) if(mixed)vc$vcov[vc$grp==group][1] else NA_real_
 diagnostics[[key]] <- data.frame(outcome=outcome,stage=stage,
  n_admissions=nrow(admissions),n_observations=nrow(d),n_pairs=nlevels(d$matched_pair),
  n_patients=nlevels(d$patient),converged=converged,
  singular=singular,numeric_valid=valid,warning_flag=warning_flag,
  patient_variance=variance("patient"),pair_variance=variance("matched_pair"),
  residual_variance=variance("Residual"),covariance_min_eigenvalue=min(eigenvalues),
  warnings=paste(c(warnings,messages),collapse=" | "),
  framework=if(mixed)"REML linear mixed model" else "PPML; two-way patient/pair HC1 SE")
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
  if(mixed) {
   test <- contest1D(fit,L,ddf="Satterthwaite")
   df <- test[["df"]]
  }
  cutoff <- if(mixed)qt(.975,df) else qnorm(.975)
  p <- if(valid)2*pt(-abs(beta/se),df=df) else NA_real_
  lo <- beta-cutoff*se; hi <- beta+cutoff*se
  results[[paste(key,label)]] <- data.frame(outcome=outcome,stage=stage,
   contrast=label,beta=beta,se=se,df=df,
   effect=if(mixed)beta else exp(beta),
   ci_low=if(mixed)lo else exp(lo),ci_high=if(mixed)hi else exp(hi),
   p_raw_exploratory=p,warning_flag=warning_flag,
   effect_type=if(clinical)"event rate ratio" else if(mixed)"Flesch reading ease point difference" else
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
write.csv(do.call(rbind,results),file.path(output,"group_contrasts.csv"),row.names=FALSE)
write.csv(do.call(rbind,diagnostics),file.path(output,"model_diagnostics.csv"),row.names=FALSE)
write.csv(do.call(rbind,coefficients),file.path(output,"model_coefficients.csv"),row.names=FALSE)
if(clinical)write.csv(do.call(rbind,calibration),
 file.path(output,"mean_calibration_by_group_period.csv"),row.names=FALSE)
capture.output(sessionInfo(),file=file.path(output,"R_session_info.txt"))
