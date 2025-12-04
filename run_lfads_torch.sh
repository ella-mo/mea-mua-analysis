#!/bin/bash
#SBATCH --job-name=lfads_torch_session_test
#SBATCH --output=logs/lfads_torch_session_test_%j.out
#SBATCH --error=logs/lfads_torch_session_test_%j.err
#SBATCH -p batch
#SBATCH --time=00:30:00
#SBATCH --mem=5G

# ----------------------------
# User paths (EDIT THESE)
# ----------------------------
LFADS_DIR=/oscar/data/slizarra/emohanra/finding_latent_rates_with_kilosort/lfads-torch
CONFIG_PATH="$LFADS_DIR/functions/config.yaml"

# ----------------------------
# Setup
# ----------------------------
module load miniconda3/23.11.0s
eval "$(conda shell.bash hook)"   
conda activate lfads-torch

echo "Starting LFADS-Torch single-session training..."
echo "Repository: $LFADS_DIR"
echo "Node: $(hostname)"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
echo "Time: $(date)"

echo "Python path: $(which python)"
python -c "import sys; print('sys.path:', sys.path)"
python -c "import site; print('site-packages:', site.getsitepackages())"
python -c "import hydra; print('Hydra version:', hydra.__version__)"


mkdir -p logs

# ----------------------------
# Run LFADS single-session
# ----------------------------
cd "$LFADS_DIR"

# preprocess all bin files and build configs
echo "=========================================="
echo "Starting preprocessing of bin files..."
echo "Reading bin files from: bin_files.csv"
echo "Current directory: $(pwd)"
echo "Verifying bin_files.csv exists:"
if [ -f "bin_files.csv" ]; then
    echo "  ✓ bin_files.csv found"
    # Count non-empty lines after header (handles files without trailing newline)
    FILE_COUNT=$(tail -n +2 bin_files.csv | grep -c . || echo "0")
    echo "  Number of files to process: $FILE_COUNT"
    echo "  Files:"
    tail -n +2 bin_files.csv | while IFS= read -r line || [ -n "$line" ]; do
        if [ -n "$line" ]; then
            echo "    - $line"
        fi
    done
else
    echo "  ✗ ERROR: bin_files.csv not found!"
    exit 1
fi
echo "=========================================="
python -u -m functions.main -b bin_files.csv -l "$LFADS_DIR" -c "$CONFIG_PATH"
if [ $? -ne 0 ]; then
    echo "ERROR: Preprocessing failed with exit code $?"
    exit 1
fi
echo "=========================================="
echo "Preprocessing completed successfully"
echo "=========================================="

# grab the dataset IDs the Python step would have generated
mapfile -t DATASETS < <(
  CONFIG_PATH="$CONFIG_PATH" python - <<'PY'
import os
import pandas as pd
from pathlib import Path
from functions.making_names import make_dataset_str
import yaml
config_path = os.environ['CONFIG_PATH']
with open(config_path, "r") as f:
    config = yaml.safe_load(f)
paths = pd.read_csv("bin_files.csv")["path"]
for p in paths:
    print(make_dataset_str(Path(p), config['make_data']['bin_size'], config['make_data']['sample_len'], config['make_data']['overlap']))
PY
)

# launch LFADS for each dataset
for dataset in "${DATASETS[@]}"; do
  echo "Running LFADS for $dataset"
  sbatch -p gpu --gres=gpu:1 --time=06:00:00 --mem=10G \
    --mail-type=ALL --mail-user=ella_mohanram@brown.edu \
    -o "logs/${dataset}_%j.out" -e "logs/${dataset}_%j.err" \
    --wrap "module load miniconda3/23.11.0s && eval \"\$(conda shell.bash hook)\" && conda activate lfads-torch && cd $LFADS_DIR && python -m scripts.run_test -d \"$dataset\""
done
