#!/usr/bin/env bash
# Run from an existing ML environment on the approved computing cluster.
# Usage: bash run_current_cohort_inference.sh LOS_TRAINING_OUTPUT_DIR READMISSION_TRAINING_OUTPUT_DIR
set -euo pipefail
if [[ $# -ne 2 ]]; then
    echo "Usage: bash $0 LOS_TRAINING_OUTPUT_DIR READMISSION_TRAINING_OUTPUT_DIR" >&2
    exit 1
fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
los_dir="$1"
readmission_dir="$2"
python_bin="${INFERENCE_PYTHON:-python}"
dataset="$script_dir/test_fairness_mhc1_mhc0.parquet"
if [[ ! -f "$dataset" ]]; then
    dataset="$script_dir/01_training_data_creation/prediction_model_dataset_current_cohort_test/test_fairness_mhc1_mhc0.parquet"
fi
for required in "$dataset" "$los_dir/run_config.json" "$readmission_dir/run_config.json"; do
    if [[ ! -f "$required" ]]; then
        echo "Missing required file: $required" >&2
        exit 1
    fi
done
# Avoid carrying a smoke-test row cap or LOS filtering into this evaluation.
unset LOS_INFERENCE_MAX_ROWS LOS_FILTER_COLUMN
common_args=(--skip-general-test --test-fairness-path "$dataset"
    --model-name emilyalsentzer/Bio_ClinicalBERT --text-column model_text
    --max-length 512 --max-chunks 4 --pooling-strategy mean_max --batch-size 16 --fp16)
"$python_bin" "$script_dir/04_run_admission_classifier_inference.py" \
    "${common_args[@]}" --model-dir "$los_dir/best_model" --training-output-dir "$los_dir" \
    --label-column prolonged_los_gt_7d --threshold 0.25 \
    --output-dir "$script_dir/bioclinicalbert_los_classifier_inference_output_current_cohort"
"$python_bin" "$script_dir/04_run_admission_classifier_inference.py" \
    "${common_args[@]}" --model-dir "$readmission_dir/best_model" --training-output-dir "$readmission_dir" \
    --label-column readmission_within_30d --filter-column eligible_for_30d_readmission --threshold 0.20 \
    --output-dir "$script_dir/bioclinicalbert_readmission_classifier_inference_output_current_cohort"
