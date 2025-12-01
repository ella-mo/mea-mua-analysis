#!/bin/bash
#SBATCH --job-name=toy_session_test
#SBATCH --output=logs/toy_session_test_%j.out
#SBATCH --error=logs/toy_session_test_%j.err
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --mem=10G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ella_mohanram@brown.edu


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

echo "Starting LFADS-Torch toy session training..."
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
# Run LFADS toy session
# ----------------------------

cd "$LFADS_DIR"

# preprocess all sample sizes and build configs
python -m functions.toy_data_main -l "$LFADS_DIR" -c "$CONFIG_PATH"

# grab the toy dataset IDs that toy_data_main just generated
mapfile -t rate_str_ss_lst < <(python - <<'PY'
from pathlib import Path
from functions.making_names import make_toy_dataset_strs

config_path = Path("$CONFIG_PATH")
_, rate_str_ss_lst = make_toy_dataset_strs(config_path)

for s in rate_str_ss_lst:
    print(s)
PY
)

# launch LFADS for each toy dataset
for rate_str_ss in "${rate_str_ss_lst[@]}"; do
  echo "Running LFADS for $rate_str_ss"
  python -m scripts.run_test -d "$rate_str_ss"
done

echo "Training complete!"
echo "Time: $(date)"