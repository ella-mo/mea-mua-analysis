# run toy data through lfads to see how encod_seq_len changes 
import numpy as np 
import h5py
import sys
from pathlib import Path
import yaml
import os
from typing import Optional
from scipy.io import savemat

# Handle imports for both script execution and module import
try:
    # Try relative imports first (when imported as module)
    from .make_configs import create_datamodule_config, create_model_config
    from .bin_data import bin_make_train_val, readable_float
    from .process_data import lambda_t, inhomogeneous_poisson_sinusoidal
    from .making_names import make_toy_dataset_strs
except ImportError:
    # If relative imports fail, add parent directory to path and use absolute imports
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from functions.make_configs import create_datamodule_config, create_model_config
    from functions.bin_data import bin_make_train_val, readable_float
    from functions.process_data import lambda_t, inhomogeneous_poisson_sinusoidal
    from functions.making_names import make_toy_dataset_strs

if __name__ == '__main__':
    # hardcoded
    lfads_dir = Path(__file__).resolve().parent.parent
    config_path = lfads_dir / 'functions/config.yaml'

    import argparse
    parser = argparse.ArgumentParser(description="Run LFADS on toy data")
    parser.add_argument("-l", "--lfads_dir", type=str, default=lfads_dir, help="full file path to lfads-torch directory")
    parser.add_argument("-c", "--config_path", type=str, default=config_path, help="full file path to config file")
    args = parser.parse_args()

    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

    out_path = args.lfads_dir
    original_cwd = Path.cwd()

    n_channels = int(config['make_toy_data']['num_channels'])
    DEBUG = config['make_toy_data']['DEBUG']
    sample_sizes = config['make_toy_data']['sample_sizes']
    bin_size = config['make_toy_data']['bin_size']
    overlaps = [(sample_sizes[i]-sample_sizes[0]) * bin_size for i in range(len(sample_sizes))]
    recording_duration = float(config['make_toy_data']['recording_duration'])
    max_rate = float(config['make_toy_data']['max_rate'])
    min_rate = float(config['make_toy_data']['min_rate'])
    period = float(config['make_toy_data']['period'])
    phase = float(config['make_toy_data'].get('phase', 0))
    start_sin_time = float(config['make_toy_data']['start_sin_time'])
    end_sin_time = float(config['make_toy_data']['end_sin_time'])

    rate_str, rate_str_ss_lst = make_toy_datset_strs(args.config_path)
    files_dir = lfads_dir / "files" / rate_str
    files_dir.mkdir(parents=True, exist_ok=True)
    
    # Simulate the process
    spike_times_per_channel = []
    rate_f = lambda_t(max_rate, min_rate, period, start_sin_time, end_sin_time)
    for ch in range(n_channels):
        spike_times = inhomogeneous_poisson_sinusoidal(recording_duration, rate_f, max_rate)
        spike_times_per_channel.append(spike_times)
        print(f'Channel {ch}: {len(spike_times)} spikes')

    # Plot the results
    if DEBUG:
        import matplotlib.pyplot as plt
        for ch in range(n_channels):
            spike_times = spike_times_per_channel[ch]
            times = np.linspace(0, recording_duration, 1000)

            plt.figure(figsize=(10, 5))
            plt.plot(times, rate_f(times), label='$\\lambda(t)$', color='blue')
            plt.scatter(spike_times, np.zeros_like(spike_times), color='red', marker='|', s=100, label=f'Spikes ({len(spike_times)} total)')
            plt.title('Inhomogeneous Poisson Process with Sinusoidal Intensity')
            plt.xlabel('Time (s)')
            plt.ylabel('Rate $\\lambda(t)$ (Hz)')
            plt.legend()
            plt.grid(True, axis='y', linestyle='--', alpha=0.6)
            plt.show()
        
        sys.exit()
    
    np.save(files_dir / f'{rate_str}_data.npy', 
        {index: ch_times for index, ch_times in enumerate(spike_times_per_channel)},
        allow_pickle=True)
    print('Saved data in files/')

    split_frac = float(config['make_toy_data']['split_frac'])
    batch_size = int(config['make_toy_data']['batch_size'])

    for i, (sample_size, overlap, rate_str_ss) in enumerate(zip(sample_sizes, overlaps, rate_str_ss_lst)):
        #Prep output paths
        train_indices = f"{lfads_dir}/files/train_indices_{rate_str_ss}.npy"
        valid_indices = f"{lfads_dir}/files/valid_indices_{rate_str_ss}.npy"
        data_file = f'{out_path}/datasets/{rate_str_ss}.h5'

        print(f'Prepping files for {sample_size} sample size ({overlap} overlap)')
        print('\n')

        # sample_size is in bins, convert to seconds for sample_len
        sample_len = sample_size * bin_size
        train_data, valid_data, train_idx, valid_idx = bin_make_train_val(
            spike_times_per_channel, 
            n_channels, 
            recording_duration, 
            sample_len, 
            bin_size, 
            overlap, 
            split_frac, 
            DEBUG
        )

        # Save index lists for later reconstruction
        np.save(train_indices, train_idx)
        np.save(valid_indices, valid_idx)

        print(f"Saved index lists for reconstruction.")

        with h5py.File(data_file, "w") as f:
            f.create_dataset("train_encod_data", data=train_data)
            f.create_dataset("train_recon_data", data=train_data)
            f.create_dataset("valid_encod_data", data=valid_data)
            f.create_dataset("valid_recon_data", data=valid_data)

        # # config files
        create_datamodule_config(out_path, batch_size, rate_str_ss)
        create_model_config(out_path, train_data, rate_str_ss)
