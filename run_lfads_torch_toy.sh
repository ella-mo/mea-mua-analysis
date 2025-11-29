#!/bin/bash
#SBATCH --job-name=toy_session_test
#SBATCH --output=logs/toy_session_test_%j.out
#SBATCH --error=logs/toy_session_test_%j.err
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --mem=25G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ella_mohanram@brown.edu


# ----------------------------
# User paths (EDIT THESE)
# ----------------------------
LFADS_DIR=/oscar/data/slizarra/emohanra/finding_latent_rates_with_kilosort/lfads-torch

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
python -m functions.toy_data_main -l "$LFADS_DIR" -c functions/config.yaml

# grab the toy dataset IDs that toy_data_main just generated
mapfile -t DATASETS < <(
python - <<'PY'
import yaml
from pathlib import Path
from functions.bin_data import readable_float

lfads_dir = Path(".")
config_path = lfads_dir / "functions" / "config.yaml"

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

sample_sizes = config["make_toy_data"]["sample_sizes"]
max_rate = float(config["make_toy_data"]["max_rate"])
min_rate = float(config["make_toy_data"]["min_rate"])
period = float(config["make_toy_data"]["period"])

rate_str = f"toy_max{readable_float(max_rate)}_min{readable_float(min_rate)}_per{readable_float(period)}"

for sample_size in sample_sizes:
    rate_str_ss = f"{rate_str}_ss{readable_float(sample_size)}"
    print(rate_str_ss)
PY
)

# launch LFADS for each toy dataset
for dataset in "${DATASETS[@]}"; do
  echo "Running LFADS for $dataset"
  python -m scripts.run_test -d "$dataset"
done

echo "Training complete!"
echo "Time: $(date)"
