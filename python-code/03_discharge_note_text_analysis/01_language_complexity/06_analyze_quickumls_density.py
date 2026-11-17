"""Medical-term word coverage in all matched notes and three prose sections.

Local processing only. Outputs contain IDs and numeric measures, never text,
matched expressions or concepts. Checkpoint CSV permits safe interruption/resume.
This measures terminology coverage, not patient familiarity or comprehension.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import re
import time

import numpy as np
import pandas as pd
from quickumls import QuickUMLS

HERE = Path(__file__).resolve().parent
OUT = HERE / 'analysis_output_quickumls_density'
# Clinically focused types from the existing CC extractor; exclude generic
# Event/Activity and include laboratory findings and organisms for note content.
SEMTYPES = {'T184','T033','T047','T046','T048','T191','T037','T190',
            'T023','T029','T030','T038','T039','T040','T060','T061','T059',
            'T121','T200','T109','T123','T034','T007'}
WORD = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)?")


def coverage(text, matcher):
    # ASCII alphabetic words: consistent denominator; numbers/placeholders excluded.
    words = list(WORD.finditer(text))
    if not words:
        return dict(n_words=0, n_covered_words=0, n_term_spans=0, density_pct=np.nan)
    # Explicit _match avoids local QuickUMLS.match ignoring its best_match argument.
    matches = matcher._match(matcher.nlp(text), best_match=False, ignore_syntax=False)
    spans = sorted({(int(c['start']), int(c['end'])) for group in matches for c in group})
    merged = []
    for start, end in spans:
        if not 0 <= start < end <= len(text):
            raise ValueError('Invalid matcher character offsets')
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    count = 0
    j = 0
    for word in words:
        while j < len(merged) and merged[j][1] <= word.start():
            j += 1
        if j < len(merged) and merged[j][0] < word.end():
            count += 1
    assert 0 <= count <= len(words)
    return dict(n_words=len(words), n_covered_words=count,
                n_term_spans=len(spans), density_pct=100 * count / len(words))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, help='Process at most this many new notes (resumable)')
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('complexity', HERE / '01_analyze_discharge_note_language_complexity.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    OUT.mkdir(exist_ok=True)
    settings = dict(index=str(Path(os.environ.get('QUICKUMLS_INDEX_DIR', Path.home() / 'Downloads/thesis/quickumls_index_2026AA'))),
                    threshold=1.0, window=5, similarity='jaccard', semantic_types=sorted(SEMTYPES),
                    input_hashes={c['cohort']: hashlib.sha256(c['path'].read_bytes()).hexdigest() for c in module.FULL_NOTE_FILES},
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    manifest = OUT / 'settings.json'
    if manifest.exists() and json.loads(manifest.read_text()) != settings:
        raise ValueError('Configuration/input changed; preserve checkpoint and choose a fresh output directory')
    manifest.write_text(json.dumps(settings, indent=2))
    matcher = QuickUMLS(settings['index'], threshold=1.0, window=5,
                        similarity_name='jaccard', accepted_semtypes=SEMTYPES)
    matcher.nlp.max_length = 5_000_000
    notes = module.load_notes()
    if notes.duplicated(['cohort','hadm_id']).any():
        raise ValueError('Expected one note per matched admission')
    path = OUT / 'note_section_density.csv'
    previous = pd.read_csv(path) if path.exists() else pd.DataFrame()
    done = set(zip(previous.cohort, previous.hadm_id)) if len(previous) else set()
    processed = 0; started = time.monotonic()
    for row in notes.itertuples(index=False):
        if (row.cohort, row.hadm_id) in done:
            continue
        rows = []
        for section in ['full_note_text', *module.SELECTED_PROSE_SECTION_COLUMNS]:
            text = module.clean_text(getattr(row, section))
            rows.append(dict(cohort=row.cohort, subject_id=row.subject_id, hadm_id=row.hadm_id,
                             section=section, **coverage(text, matcher)))
        pd.DataFrame(rows).to_csv(path, mode='a', header=not path.exists(), index=False)
        processed += 1
        if processed % 10 == 0:
            print(f'Processed {len(done)+processed}/{len(notes)} notes; {time.monotonic()-started:.0f}s this run', flush=True)
        if args.limit and processed >= args.limit:
            break
    result = pd.read_csv(path)
    summaries = []
    for (cohort, section), g in result.groupby(['cohort','section']):
        x = g.density_pct.dropna()
        summaries.append(dict(cohort=cohort, section=section, n_admissions=len(g), n_nonempty=len(x),
                              n_subjects=g.subject_id.nunique(), mean=x.mean(), sd=x.std(),
                              median=x.median(), q1=x.quantile(.25), q3=x.quantile(.75),
                              aggregate_density_pct=100*g.n_covered_words.sum()/g.n_words.sum()))
    pd.DataFrame(summaries).to_csv(OUT/'density_summary.csv',index=False)
    complete = result.loc[result.section.eq('full_note_text')].shape[0] == len(notes)
    (OUT/'status.json').write_text(json.dumps(dict(complete=complete, expected_notes=len(notes),
        processed_notes=int(result.section.eq('full_note_text').sum())),indent=2))
    (OUT/'methods.txt').write_text(
        'All matched full notes (original text, including headings/CC) and present illness, brief hospital course, discharge instructions.\n'
        'QuickUMLS index 2026AA; Jaccard threshold 1.0, window 5, syntax heuristics enabled.\n'
        'Clinically focused semantic types are listed in settings.json. These differ from the CC selection filter.\n'
        'Density = 100 * alphabetic word occurrences overlapping any accepted term span / all alphabetic word occurrences.\n'
        'Overlapping spans and alternative CUIs do not multiply word coverage; repetitions at different positions count.\n'
        'Missing/wordless sections have undefined density, not zero. Numbers and deidentification underscores are excluded.\n'
        'Per-note means weight notes equally; aggregate density weights by alphabetic word count.\n'
        'This is dictionary-defined medical terminology coverage, not a validated jargon/comprehension score.\n'
        'Descriptive analysis only; no group significance tests or additional model adjustments.\n')
    print(f'Complete: {complete}. Numeric outputs: {OUT}', flush=True)


def test_density():
    """OLS M0/M2 with patient/pair HC1 covariance; four-scope BH per stage."""
    import subprocess
    if not json.loads((OUT / 'status.json').read_text())['complete']:
        raise ValueError('Extraction must finish before testing')
    r = r'''
args <- commandArgs(trailingOnly=TRUE)
.libPaths(c(file.path(args[1],'.r-library'),.libPaths()))
library(sandwich)
out <- args[2]
x <- read.csv(file.path(out,'note_section_density.csv'))
cov <- read.csv(file.path(args[1],'analysis_output_language_complexity_inference',
                         'race_ethnicity_sensitivity','model_covariates.csv'))
keys <- c('cohort','subject_id','hadm_id')
stopifnot(!anyDuplicated(cov[,keys]))
x <- merge(x,cov,by=keys,all.x=TRUE,sort=FALSE)
stopifnot(nrow(x)==12864*4,!anyNA(x$pair_id))
models <- list(M0='mhc1', M2=c('mhc1','age_at_admission_per_10y',
                     'elixhauser_score_per_5pt','log1p_prior_all_admissions'))
results <- list(); descriptives <- list()
for(scope in unique(x$section)) {
 d <- x[x$section==scope,]
 required <- c('density_pct','pair_id','subject_id',models$M2[-1])
 valid <- complete.cases(d[,required])
 keep <- tapply(valid,d$pair_id,all) & table(d$pair_id)==2
 d <- d[d$pair_id %in% names(keep)[keep],]
 stopifnot(all(table(d$pair_id)==2))
 d$mhc1 <- as.integer(d$cohort=='MHC1_psychotic')
 stopifnot(all(tapply(d$mhc1,d$pair_id,sum)==1))
 for(cohort in unique(d$cohort)) {
  z <- d[d$cohort==cohort,]
  descriptives[[paste(scope,cohort)]] <- data.frame(section=scope,cohort,
   n_admissions=nrow(z),n_patients=length(unique(z$subject_id)),
   mean=mean(z$density_pct),sd=sd(z$density_pct),median=median(z$density_pct))
 }
 for(stage in names(models)) {
  fit <- lm(reformulate(models[[stage]],response='density_pct'),data=d)
  stopifnot(fit$rank==length(coef(fit)))
  v <- vcovCL(fit,cluster=list(d$subject_id,d$pair_id),type='HC1',cadjust=TRUE,fix=FALSE)
  eig <- eigen(v,symmetric=TRUE,only.values=TRUE)$values
  stopifnot(min(eig)> -1e-8*max(1,max(abs(eig))),v['mhc1','mhc1']>0)
  b <- coef(fit)['mhc1']; se <- sqrt(v['mhc1','mhc1'])
  results[[paste(scope,stage)]] <- data.frame(section=scope,stage,
   n_pairs=length(unique(d$pair_id)),difference_pp=b,se,
   ci_low=b-qnorm(.975)*se,ci_high=b+qnorm(.975)*se,
   p=2*pnorm(-abs(b/se)),covariance_psd=TRUE)
 }
}
r <- do.call(rbind,results)
for(stage in names(models)) { ix <- r$stage==stage; r$FDR[ix] <- p.adjust(r$p[ix],'BH',n=4) }
write.csv(r,file.path(out,'density_cohort_effects.csv'),row.names=FALSE)
write.csv(do.call(rbind,descriptives),file.path(out,'density_complete_pair_descriptives.csv'),row.names=FALSE)
writeLines(c('OLS M0 cohort only; M2 additionally age/10, Elixhauser/5, log(1+prior all-MIMIC admissions).',
 'Patient and matched-pair two-way HC1 clustered covariance, finite cluster corrections, no covariance repair.',
 'Outcome-specific complete pairs; same sample for M0/M2. Normal Wald tests and pointwise 95% confidence intervals.',
 'BH separately across four text scopes within M0 and within M2. Exploratory analysis added after original readability results.',
 'Effect is MHC1 minus MHC0 in percentage points of alphabetic word coverage, not relative percent difference.'),
 file.path(out,'density_inference_methods.txt'))
print(r,row.names=FALSE)
'''
    subprocess.run(['/usr/local/bin/Rscript','-',str(HERE),str(OUT)],input=r,text=True,check=True)


if __name__ == '__main__':
    import sys
    if '--test-density' in sys.argv:
        test_density()
    else:
        main()
