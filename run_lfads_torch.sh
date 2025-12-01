#!/bin/bash
#SBATCH --job-name=session_test
#SBATCH --output=logs/session_test_%j.out
#SBATCH --error=logs/session_test_%j.err
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
#!/bin/bash
# ... existing Slurm + conda setup ...

cd "$LFADS_DIR"

# preprocess all bin files and build configs
python -m functions.main -b bin_files.csv -l "$LFADS_DIR" -c functions/config.yaml

# grab the dataset IDs the Python step would have generated
mapfile -t DATASETS < <(
  python - <<'PY'
import pandas as pd
from pathlib import Path
from functions.making_names import make_dataset_str
import yaml
with open("functions/config.yaml", "r") as f:
    config = yaml.safe_load(f)
paths = pd.read_csv("bin_files.csv")["path"]
for p in paths:
    print(make_dataset_str(Path(p), config['make_data']['bin_size'], config['make_data']['sample_len'], config['make_data']['overlap']))
PY
)

# launch LFADS for each dataset
for dataset in "${DATASETS[@]}"; do
  echo "Running LFADS for $dataset"
  python -m scripts.run_test -d "$dataset"
done

echo "Training complete!"
echo "Time: $(date)"
