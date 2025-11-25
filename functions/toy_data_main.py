# run toy data through lfads to see how encod_seq_len changes 
import numpy as np 
import h5py
import sys
import math
from pathlib import Path
import yaml
import os
from typing import Optional
from scipy.io import savemat

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
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))
    from data_functions import stitch_data

def readable_float(float_string):
    return str(float_string).replace('.','_')

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

def inhomogeneous_poisson_sinusoidal(
    duration: float,
    max_rate: float,
    min_rate: float,
    frequency: float,
    phase: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Simulate an inhomogeneous Poisson process with sinusoidal rate via thinning.

    Args:
        duration: Total simulation time (seconds).
        max_rate: Maximum rate (Hz). Defines the rejection envelope.
        min_rate: Minimum rate (Hz). Must satisfy 0 <= min_rate <= max_rate.
        frequency: Sinusoid frequency (Hz).
        phase: Optional phase offset (radians).
        rng: Optional numpy Generator for reproducibility.

    Returns:
        np.ndarray of event times (seconds) sorted in ascending order.
    """
    if max_rate <= 0:
        raise ValueError("max_rate must be positive.")
    if min_rate < 0 or min_rate > max_rate:
        raise ValueError("min_rate must be in [0, max_rate].")
    if frequency <= 0:
        raise ValueError("frequency must be positive.")
    if rng is None:
        rng = np.random.default_rng()

    def lambda_t(t: float) -> float:
        # Sinusoid scaled to [min_rate, max_rate]
        return min_rate + (max_rate - min_rate) * 0.5 * (1 + math.sin(2 * math.pi * frequency * t + phase))

    lam_max = max_rate
    t = 0.0
    events = []
    while t < duration:
        t += rng.exponential(1.0 / lam_max)
        if t >= duration:
            break
        if rng.random() < lambda_t(t) / lam_max:
            events.append(t)
    return np.array(events, dtype=float)


if __name__ == '__main__':
    lfads_dir = Path(__file__).resolve().parent.parent
    config_file = lfads_dir / 'functions/config.yaml'

    import argparse
    parser = argparse.ArgumentParser(description="Run LFADS on toy data")
    parser.add_argument("-l", "--lfads_dir", type=str, default=lfads_dir, help="full file path to lfads-torch directory")
    parser.add_argument("-c", "--config_file", type=str, default=config_file, help="full file path to config file")
    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)

    out_path = args.lfads_dir
    original_cwd = Path.cwd()
    files_dir = lfads_dir / "files"
    files_dir.mkdir(parents=True, exist_ok=True)

    n_channels = int(config['make_toy_data']['num_channels'])
    DEBUG = config['make_toy_data']['DEBUG']
    # YAML automatically parses lists, so we can use it directly
    sample_sizes = config['make_toy_data']['sample_sizes']
    bin_size = config['make_toy_data']['bin_size']
    # Convert sample size differences to overlap in seconds
    overlaps = [(sample_sizes[i]-sample_sizes[0]) * bin_size for i in range(len(sample_sizes))]
    # Parameters
    recording_duration = float(config['make_toy_data']['recording_duration'])
    max_rate = float(config['make_toy_data']['max_rate'])
    min_rate = float(config['make_toy_data']['min_rate'])
    frequency = float(config['make_toy_data']['frequency'])
    phase = float(config['make_toy_data'].get('phase', 0))

    rate_str = f"toy_max{readable_float(max_rate)}_min{readable_float(min_rate)}_freq{readable_float(frequency)}"
    
    # Simulate the process
    spike_times_per_channel = []
    for ch in range(n_channels):
        spike_times = inhomogeneous_poisson_sinusoidal(recording_duration, max_rate, min_rate, frequency)
        spike_times_per_channel.append(spike_times)
        print(f'Channel {ch}: {len(spike_times)} spikes')

    # Plot the results
    if DEBUG:
        import matplotlib.pyplot as plt
        for ch in range(n_channels):
            spike_times = spike_times_per_channel[ch]
            times = np.linspace(0, recording_duration, 1000)
            rates = min_rate + (max_rate - min_rate) * 0.5 * (1 + np.sin(2 * np.pi * frequency * times + phase))

            plt.figure(figsize=(10, 5))
            plt.plot(times, rates, label='$\\lambda(t)$', color='blue')
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

    for i, (sample_size, overlap) in enumerate(zip(sample_sizes, overlaps)):
        #Prep output paths
        train_indices = f"{lfads_dir}/files/train_indices_{rate_str}.npy"
        valid_indices = f"{lfads_dir}/files/valid_indices_{rate_str}.npy"
        data_file = f'{out_path}/datasets/{rate_str}.h5'
        runs_dir = Path(out_path) / "runs" / rate_str

        if config['make_toy_data']['run_lfads']:
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
            create_datamodule_config(out_path, batch_size, rate_str)
            create_model_config(out_path, rate_str, train_data)

            sys.exit()

            import subprocess
            import sys

            env_name = "lfads-torch" 
            script_to_run = "scripts/run_test.py"

            os.chdir(out_path)
            print(f'Currently in {os.getcwd()}')
            try:
                # Construct the command to run the script within the specified Conda environment
                command = [
                    "conda",
                    "run",
                    "-n",
                    env_name,
                    "python",  # Use the Python interpreter from the lfads-torch Conda env
                    script_to_run,
                    "-d",
                    dataset_str,
                ]

                # Execute the command using subprocess.run
                # capture_output=True to capture stdout and stderr
                # text=True to decode output as text
                # check=True to raise an exception if the command returns a non-zero exit code
                result = subprocess.run(command, capture_output=True, text=True, check=True)

                print("Script output:")
                print(result.stdout)
                if result.stderr:
                    print("Script errors:")
                    print(result.stderr)

            except subprocess.CalledProcessError as e:
                print(f"Error running script in Conda environment: {e}")
                print(f"Stdout: {e.stdout}")
                print(f"Stderr: {e.stderr}")
            except FileNotFoundError:
                print("Error: 'conda' command not found. Ensure Conda is installed and in your PATH.")
            finally:
                os.chdir(original_cwd)


        latest_rates_file = get_latest_lfads_output(runs_dir)
        if config['make_toy_data']['stitch_data']:
            if latest_rates_file is None:
                if config['make_toy_data']['run_lfads']:
                    raise FileNotFoundError(
                        f"Could not locate LFADS output in '{runs_dir}'."
                    )
                else:
                    raise FileNotFoundError(
                        f"run_lfads is False but no prior LFADS output was found in '{runs_dir}'."
                    )

            stitch_data(
                latest_rates_file,
                "rates",
                train_indices,
                valid_indices,
                bin_size,
                overlap,
                files_dir,
            )
