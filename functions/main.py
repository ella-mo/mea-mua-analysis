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
    from .bin_data import bin_make_train_val, readable_float
    from .process_data import extract_threshold_waveforms, calculate_threshold
    from .making_names import make_dataset_str

except ImportError:
    # If relative imports fail, add parent directory to path and use absolute imports
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from functions.make_configs import create_datamodule_config, create_model_config
    from functions.bin_data import bin_make_train_val, readable_float
    _project_root = Path(__file__).parent.parent.parent
    from data_functions import stitch_data
    from process_data import extract_threshold_waveforms, calculate_threshold
    from making_names import make_dataset_str

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Run LFADS on multiple bin files")
    parser.add_argument("-b", "--bin_files_csv", type=str, help="full path to .csv file of the .bin file paths to run LFADS on, column should be called be path, each line should be a full path to bin file")
    parser.add_argument("-l", "--lfads_dir", type=str, help="full file path to lfads-torch directory")
    parser.add_argument("-c", "--config_file", type=str, help="full file path to config file")
    args = parser.parse_args()

    # process args
    files = pd.read_csv(args.bin_files_csv)['path'].tolist()

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
        dataset_str = make_dataset_str(file, bin_size, sample_len, overlap)
        os.makedirs(f'{output_dir}/other_files', exist_ok=True)
        os.makedirs(f'{output_dir}/other_files/{dataset_str}', exist_ok=True)
        train_indices = f"{output_dir}/other_files/{dataset_str}/train_indices_{dataset_str}.npy"
        valid_indices = f"{output_dir}/other_files/{dataset_str}/valid_indices_{dataset_str}.npy"
        mat_file = f"{output_dir}/other_files/{dataset_str}/{dataset_str}_raw_data.mat"
                # Create datasets directory if it doesn't exist
        data_file = f'{args.lfads_dir}/datasets/{dataset_str}.h5'

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
        
        with h5py.File(data_file, "w") as f:
            f.create_dataset("train_encod_data", data=train_data)
            f.create_dataset("train_recon_data", data=train_data)
            f.create_dataset("valid_encod_data", data=valid_data)
            f.create_dataset("valid_recon_data", data=valid_data)
        
        # Get batch_size from config or use default
        batch_size = int(config['make_data'].get('batch_size', 32))
        
        create_datamodule_config(args.lfads_dir, batch_size, dataset_str)
        create_model_config(args.lfads_dir, dataset_str, train_data)