from pathlib import Path
import re
from typing import Optional
import yaml

try:
    # Try relative imports first (when imported as module)
    from .bin_data import readable_float
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from functions.bin_data import readable_float

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
def make_toy_dataset_strs(config_path):
    config_toy_str = 'make_toy_data'
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    sample_sizes = config[config_toy_str]["sample_sizes"]
    max_rate = float(config[config_toy_str]["max_rate"])
    min_rate = float(config[config_toy_str]["min_rate"])
    period = float(config[config_toy_str]["period"])

    rate_str = f"toy_max{readable_float(max_rate)}_min{readable_float(min_rate)}_per{readable_float(period)}"

    rate_str_ss_lst = [f"{rate_str}_ss{readable_float(sample_size)}" for sample_size in sample_sizes]

    return rate_str, rate_str_ss_lst
