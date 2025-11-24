import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lfads_torch.run_model import run_model

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run LFADS on toy data")
    parser.add_argument("-d", "--dataset_str", type=str, 
        help="dataset_str")
    parser.add_argument("-o", "--overwrite", type=bool, default=True)
    args = parser.parse_args()

    DATASET_STR = args.dataset_str
    OVERWRITE = args.overwrite
    RUN_TAG = datetime.now().strftime("%y%m%d%H%M") + "_exampleSingle"
    RUN_DIR = Path("runs") / DATASET_STR / RUN_TAG

    # Overwrite the directory if necessary
    if RUN_DIR.exists() and OVERWRITE:
        shutil.rmtree(RUN_DIR)
    RUN_DIR.mkdir(parents=True)
    # Copy this script into the run directory
    shutil.copyfile(__file__, RUN_DIR / Path(__file__).name)
    # Switch to the `RUN_DIR` and train the model
    os.chdir(RUN_DIR)
    run_model(
        overrides={
            "datamodule": DATASET_STR,
            "model": DATASET_STR,
        },
        config_path="../configs/single.yaml",
        checkpoint_dir=None
    )
