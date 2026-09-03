#!/bin/bash
#SBATCH -J LiCseng_LOS_BERT
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --partition=a100-galvani
#SBATCH --time=0-08:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=logs/los-bert-%j.out
#SBATCH --error=logs/los-bert-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=csenge.frater@student.uni-tuebingen.de

# Diagnostic and Analysis Phase - please leave these in.
scontrol show job $SLURM_JOB_ID
pwd
nvidia-smi
ls $WORK

# Activate your cluster Python environment.
source "$HOME/thesis_env/bin/activate"

echo "Python executable after activating env:"
which python
python --version

python -c 'import sys; print("Python version info:", sys.version_info); sys.exit(0 if sys.version_info >= (3, 10) else 1)'

# Keep Hugging Face downloads/cache off slower project storage if possible.
export HF_HOME="$HOME/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"

# Bio_ClinicalBERT LOS classifier settings.
export LOS_MODEL_NAME="emilyalsentzer/Bio_ClinicalBERT"
export LOS_TEXT_COLUMN=model_text
export LOS_LABEL_COLUMN=prolonged_los_gt_7d
export LOS_MAX_LENGTH=512
export LOS_NUM_TRAIN_EPOCHS=3
export LOS_LEARNING_RATE=2e-5
export LOS_WEIGHT_DECAY=0.01
export LOS_WARMUP_RATIO=0.06
export LOS_TRAIN_BATCH_SIZE=16
export LOS_EVAL_BATCH_SIZE=32
export LOS_GRADIENT_ACCUMULATION_STEPS=1
export LOS_LOGGING_STEPS=50
export LOS_SAVE_TOTAL_LIMIT=2
export LOS_FP16=1
export LOS_BF16=0
export LOS_USE_CLASS_WEIGHTS=0
export LOS_SEED=20260903

# Uncomment these for a quick cluster smoke test.
export TRAIN_MAX_ROWS=500
08_export VALIDATION_MAX_ROWS=500

if [[ -f "02_train_bioclinicalbert_los_classifier.py" ]]; then
    LOS_SCRIPT="02_train_bioclinicalbert_los_classifier.py"
elif [[ -f "python-code/08_model_training/02_train_bioclinicalbert_los_classifier.py" ]]; then
    LOS_SCRIPT="python-code/08_model_training/02_train_bioclinicalbert_los_classifier.py"
elif [[ -f "08_model_training/02_train_bioclinicalbert_los_classifier.py" ]]; then
    LOS_SCRIPT="08_model_training/02_train_bioclinicalbert_los_classifier.py"
else
    echo "Could not find 02_train_bioclinicalbert_los_classifier.py from $(pwd)"
    exit 1
fi

LOS_SCRIPT_DIR="$(dirname "$LOS_SCRIPT")"
export LOS_TRAIN_PATH="${LOS_SCRIPT_DIR}/01_training_data_creation/prediction_model_dataset/train.parquet"
export LOS_VALIDATION_PATH="${LOS_SCRIPT_DIR}/01_training_data_creation/prediction_model_dataset/validation.parquet"
export LOS_OUTPUT_DIR="${LOS_SCRIPT_DIR}/bioclinicalbert_los_classifier_output"

mkdir -p logs
mkdir -p "$LOS_OUTPUT_DIR"

echo "Training script: $LOS_SCRIPT"
echo "Train data: $LOS_TRAIN_PATH"
echo "Validation data: $LOS_VALIDATION_PATH"
echo "Output dir: $LOS_OUTPUT_DIR"

srun python "$LOS_SCRIPT"
