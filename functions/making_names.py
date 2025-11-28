from pathlib import Path
import re
from typing import Optional


# REAL DATA
def make_dataset_str(bin_file, bin_size, sample_len, overlap):
    """Extract day, recording number, and well from bin file path and name"""
    bin_path = Path(bin_file)
    bin_name = bin_path.name
    
    # Extract well from bin filename (e.g., "20250509_NIN-B1_D80(001)_C5.bin" -> "C5")
    well_match = re.search(r'_([A-D][1-8])\.bin$', bin_name)
    if not well_match:
        raise ValueError(f"Could not extract well from bin filename: {bin_name}")
    well = well_match.group(1)
    
    # Extract day from grandparent directory (e.g., "20250509_NIN-B1_D80" or "20250410_NIN_B1_D51" -> "80" or "51")
    grandparent_dir = bin_path.parent.parent.name
    day_match = re.search(r'B1_D(\d+)', grandparent_dir)
    if not day_match:
        raise ValueError(f"Could not extract day from grandparent directory: {grandparent_dir}")
    day = day_match.group(1)
    
    # Extract recording number from bin filename (e.g., "20250509_NIN-B1_D80(001)_C5.bin" -> "001")
    rec_match = re.search(r'\((\d+)\)', bin_name)
    if not rec_match:
        raise ValueError(f"Could not extract recording number from bin filename: {bin_name}")
    recording = rec_match.group(1)
    
    dataset_str = f'd{day}_r{recording}_w{well}_b{readable_float(bin_size)}_sl{readable_float(sample_len)}_o{readable_float(overlap)}'

    return dataset_str

# TOY DATA
def get_latest_lfads_output(runs_root: Path) -> Optional[Path]:
    """
    Returns the most recent LFADS output .h5 file within runs_root (runs/dataset_str/RUN_TAG).
    """
    if not runs_root.exists():
        return None
    run_dirs = [d for d in runs_root.iterdir() if d.is_dir()]
    if not run_dirs:
        return None
    latest_run = max(run_dirs, key=lambda p: p.stat().st_mtime)
    rate_files = sorted(latest_run.glob("lfads_output*.h5"))
    if not rate_files:
        return None
    return rate_files[-1]
