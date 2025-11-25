import numpy as np
import glob as glob
from pathlib import Path
import h5py, os, re
import sys
from scipy.io import savemat
import pandas as pd
import yaml

# Handle imports for both script execution and module import
try:
    # Try relative imports first (when imported as module)
    from .make_configs import create_datamodule_config, create_model_config
    from .bin_data import bin_make_train_val
except ImportError:
    # If relative imports fail, add parent directory to path and use absolute imports
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from functions.make_configs import create_datamodule_config, create_model_config
    from functions.bin_data import bin_make_train_val
    _project_root = Path(__file__).parent.parent.parent
    from data_functions import stitch_data

def extract_threshold_waveforms(signal, threshold, fs):
    """
    Extracts spike-aligned waveforms and timing information.

    Parameters
    ----------
    signal : 1D array
        Voltage trace.
    threshold : float
        Threshold value (positive).
    fs : float
        Sampling frequency in Hz (e.g., 12500).

    Returns
    -------
    crossing_times : (num_crossings,) array
        Crossing times in seconds.
    """
    samples = int(round(0.001 * fs))       # 1 ms before and after
    window = np.arange(-samples, samples + 1)
    num_samples = len(window)

    # Detect negative threshold crossings (downward crossings from above -threshold to below -threshold)
    neg_idx = np.where(np.diff(np.concatenate([[0], signal < -threshold])) == 1)[0]

    crossings = neg_idx
    num_crossings = len(crossings)

    for i, t in enumerate(crossings):
        sample_idx = t + window
        valid = (sample_idx >= 0) & (sample_idx < len(signal))

    crossing_times = crossings / fs

    return crossing_times


def calculate_threshold(curr_channel_data):
    median_val = np.median(curr_channel_data)
    absolute_deviations = np.abs(curr_channel_data - median_val)
    mad = np.median(absolute_deviations)
    stdev = mad / 0.6745
    threshold = 4 * stdev

    return threshold


def extract_info_from_bin_file(bin_file):
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
    
    filename = f'd{day}_r{recording}_w{well}'

    return filename


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Run LFADS on multiple bin files")
    parser.add_argument("-b", "--bin_files_csv", type=str, help="full path to .csv file of the .bin file paths to run LFADS on, column should be called be path, each line should be a full path to bin file")
    parser.add_argument("-l", "--lfads_dir", type=str, help="full file path to lfads-torch directory")
    parser.add_argument("-c", "--config_file", type=str, help="full file path to config file")
    args = parser.parse_args()

    # process args
    files = pd.read_csv(args.bin_files_csv)['path'].tolist()
    lfads_datasets_path = f'{args.lfads_dir}/datasets'

    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)

    num_channels = int(config['make_data']['num_channels'])
    fs = float(config['make_data']['fs'])
    bin_size = float(config['make_data']['bin_size'])
    sample_len = float(config['make_data']['sample_len'])
    overlap = float(config['make_data']['overlap'])
    split_frac = float(config['make_data']['split_frac'])
    DEBUG = config['make_data']['DEBUG']
    recording_duration = float(config['make_data']['recording_duration'])
    print(f'{sample_len} seconds per sample')

    for file in files:
        file = Path(file)
        print(f'Processing {file}')

        #Prep output paths
        output_dir = Path.cwd()
        filename = extract_info_from_bin_file(file)
        os.makedirs(f'{output_dir}/lfads_other_files', exist_ok=True)
        os.makedirs(f'{output_dir}/lfads_other_files/{filename}', exist_ok=True)
        train_indices = f"{output_dir}/lfads_other_files/{filename}_indices/train_indices_{filename}.npy"
        valid_indices = f"{output_dir}/lfads_other_files/{filename}_indices/valid_indices_{filename}.npy"
        mat_file = f"{output_dir}/lfads_other_files/{filename}_indices/{filename}_raw_data.mat"

        # Load bin file
        data = np.memmap(file, dtype='float32', mode='r')
        data = data.reshape((num_channels, len(data)//num_channels), order='F').T #shape (num_samples, num_channels)
        savemat(mat_file, {'data': data})
        print(f'data shape: {data.shape}')

        # Extract spike times for all channels
        spike_times_per_channel = []
        for ch in range(num_channels):
            x = data[:, ch]
            threshold = calculate_threshold(x)
            spike_times = extract_threshold_waveforms(x, threshold, fs)
            spike_times_per_channel.append(spike_times)
        
        print(f'Extracted spike times for {num_channels} channels')

        train_data, valid_data, train_idx, valid_idx = bin_make_train_val(spike_times_per_channel, 
            num_channels, recording_duration, sample_len, bin_size, 
            overlap, split_frac, DEBUG)

        # Save index lists for later reconstruction
        np.save(train_indices, train_idx)
        np.save(valid_indices, valid_idx)

        print(f"Train shape: {train_data.shape}, Valid shape: {valid_data.shape}")
        print(f"Saved index lists for reconstruction.")

        # Create datasets directory if it doesn't exist
        os.makedirs(lfads_datasets_path, exist_ok=True)
        data_file = f'{lfads_datasets_path}/{filename}.h5'
        
        with h5py.File(data_file, "w") as f:
            f.create_dataset("train_encod_data", data=train_data)
            f.create_dataset("train_recon_data", data=train_data)
            f.create_dataset("valid_encod_data", data=valid_data)
            f.create_dataset("valid_recon_data", data=valid_data)
        
        # Get batch_size from config or use default
        batch_size = int(config['make_data'].get('batch_size', 32))
        
        create_datamodule_config(args.lfads_dir, batch_size, filename)
        create_model_config(args.lfads_dir, filename, train_data)
    
