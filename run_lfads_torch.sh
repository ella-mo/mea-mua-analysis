#!/bin/bash
#SBATCH --job-name=lfads_torch_session_test
#SBATCH --output=logs/lfads_torch_session_test_%j.out
#SBATCH --error=logs/lfads_torch_session_test_%j.err
#SBATCH -p batch
#SBATCH --time=01:00:00
#SBATCH --mem=5G

# NOTE: this script can process and submit jobs for 7 bin files in one hour
# ----------------------------
# User paths (EDIT THESE)
# ----------------------------
LFADS_DIR=/oscar/data/slizarra/emohanra/finding_latent_rates_with_kilosort/lfads-torch
CONFIG_PATH="$LFADS_DIR/functions/config.yaml"
BIN_FILES_CSV="bin_files.csv"

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
if [ -f "$BIN_FILES_CSV" ]; then
    echo "  ✓ bin_files.csv found"
    # Count non-empty lines after header (handles files without trailing newline)
    FILE_COUNT=$(tail -n +2 $BIN_FILES_CSV | grep -c . || echo "0")
    echo "  Number of files to process: $FILE_COUNT"
    echo "  Files:"
    tail -n +2 $BIN_FILES_CSV | while IFS= read -r line || [ -n "$line" ]; do
        if [ -n "$line" ]; then
            echo "    - $line"
        fi
    done
else
    echo "  ✗ ERROR: bin_files.csv not found!"
    exit 1
fi
echo "=========================================="

# bin_files csv column should be called be path, each line should be a full path to bin file and make the dataset string
FILE_PATHS=()
DATASETS_STRS=()
while IFS= read -r file_path && IFS= read -r dataset_str; do
  FILE_PATHS+=("$file_path")
  DATASETS_STRS+=("$dataset_str")
done < <(
  BIN_FILES_CSV="$BIN_FILES_CSV" CONFIG_PATH="$CONFIG_PATH" python - <<'PY'
import os
import pandas as pd
from pathlib import Path
from functions.making_names import make_dataset_str
import yaml

with open(os.environ['CONFIG_PATH'], "r") as f:
    config = yaml.safe_load(f)

file_paths = pd.read_csv(os.environ['BIN_FILES_CSV'])['path'].tolist()
for file_path in file_paths:
    print(file_path)
    dataset_str = make_dataset_str(Path(file_path), config['make_data']['bin_size'], config['make_data']['sample_len'], config['make_data']['overlap'])
    print(dataset_str)
PY
)

for i in "${!FILE_PATHS[@]}"; do
    file_path="${FILE_PATHS[$i]}"
    dataset_str="${DATASETS_STRS[$i]}"
    # preprocess the bin file
    echo "Preprocessing $file_path"
    python -u -m functions.main -b "$file_path" -l "$LFADS_DIR" -c "$CONFIG_PATH"
    if [ $? -ne 0 ]; then
        echo "ERROR: Preprocessing failed with exit code $?"
        exit 1
    fi
    echo "=========================================="
    echo "Preprocessing $file_path completed successfully"
    echo "=========================================="

    # launch LFADS for dataset
    echo "Running LFADS for $dataset_str"
    sbatch -p gpu --gres=gpu:1 --time=08:00:00 --mem=10G \
        --mail-type=ALL --mail-user=ella_mohanram@brown.edu \
        -o "logs/${dataset_str}_%j.out" -e "logs/${dataset_str}_%j.err" \
        --wrap "module load miniconda3/23.11.0s && eval \"\$(conda shell.bash hook)\" && conda activate lfads-torch && cd $LFADS_DIR && python -m scripts.run_test -d \"$dataset_str\""
    if [ $? -ne 0 ]; then
        echo "ERROR: LFADS failed with exit code $?"
        exit 1
    fi
    echo "=========================================="
    echo "LFADS for $dataset_str completed successfully"
    echo "=========================================="
done

echo "=========================================="
echo "Preprocessing completed successfully"
echo "=========================================="

