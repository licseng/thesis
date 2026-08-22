#!/bin/bash
#SBATCH -J LiCseng                # Job name
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1                # Ensure that all cores are on the same machine with nodes=1
#SBATCH --partition=a100-galvani   # Which partition will run your job
#SBATCH --time=0-16:00             # Allowed runtime in D-HH:MM
#SBATCH --gres=gpu:1               # (optional) Requesting type and number of GPUs
#SBATCH --mem=32G                  # Total memory pool for all cores (see also --mem-per-cpu); exceeding this number will cause your job to fail.
#SBATCH --output=logs/myjob-%j.out       # File to which STDOUT will be written - make sure this is not on $HOME
#SBATCH --error=logs/myjob-%j.err        # File to which STDERR will be written - make sure this is not on $HOME
#SBATCH --mail-type=ALL            # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=csenge.frater@student.uni-tuebingen.de   # Email to which notifications will be sent

# Diagnostic and Analysis Phase - please leave these in.
scontrol show job $SLURM_JOB_ID
pwd
nvidia-smi # only if you requested gpus
ls $WORK # not necessary just here to illustrate that $WORK is available here

# Activate your cluster Python environment.
source "$HOME/thesis_env/bin/activate"

echo "Python executable after activating env:"
which python
python --version
echo "python3 executable on this node:"
which python3
python3 --version

python -c 'import sys; print("Python version info:", sys.version_info); sys.exit(0 if sys.version_info >= (3, 10) else 1)'

# Optional: keep Hugging Face downloads/cache in your home or scratch folder.
# Edit these paths if your cluster recommends using scratch storage.
export HF_HOME="$HOME/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"

# Sentiment classifier settings.
export SENTIMENT_BACKEND=transformers
export SENTIMENT_PROMPT_VERSION=A
export SENTIMENT_HF_MODEL_ID="Qwen/Qwen3-8B"
export SENTIMENT_HF_DEVICE_MAP=auto
export SENTIMENT_HF_TORCH_DTYPE=float16
export SENTIMENT_MAX_NOTES=none
export SENTIMENT_CHUNK_WORDS=800
export SENTIMENT_CHUNK_OVERLAP_WORDS=75
export SENTIMENT_MAX_NEW_TOKENS=192
export SENTIMENT_BATCH_SIZE=8 #best among the tested batch sizes (1, 2, 4, 8, 16) for the given GPU memory limit
export SENTIMENT_CHECKPOINT_EVERY_SECTIONS=500
export SENTIMENT_RESUME_FROM_CHECKPOINT=1
export SENTIMENT_MAX_EVIDENCE_WORDS=20
export SENTIMENT_MAX_REASON_WORDS=25
export SENTIMENT_MAX_EVIDENCE_CHARS=120
export SENTIMENT_MAX_REASON_CHARS=160

if [[ -f "01_run_sentiment_section_classifier.py" ]]; then
    SENTIMENT_SCRIPT="01_run_sentiment_section_classifier.py"
elif [[ -f "python-code/07_sentiment_analysis/01_run_sentiment_section_classifier.py" ]]; then
    SENTIMENT_SCRIPT="python-code/07_sentiment_analysis/01_run_sentiment_section_classifier.py"
elif [[ -f "07_sentiment_analysis/01_run_sentiment_section_classifier.py" ]]; then
    SENTIMENT_SCRIPT="07_sentiment_analysis/01_run_sentiment_section_classifier.py"
else
    echo "Could not find 01_run_sentiment_section_classifier.py from $(pwd)"
    exit 1
fi

SENTIMENT_SCRIPT_DIR="$(dirname "$SENTIMENT_SCRIPT")"
export SENTIMENT_OUTPUT_DIR="${SENTIMENT_SCRIPT_DIR}/sentiment_classifier_output_prompt_${SENTIMENT_PROMPT_VERSION}_all"

srun python "$SENTIMENT_SCRIPT"
