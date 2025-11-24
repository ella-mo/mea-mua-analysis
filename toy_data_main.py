# run toy data through lfads to see how encod_seq_len changes 
import numpy as np 
import h5py
import sys
import math
from pathlib import Path
import yaml
import os
from scipy.io import savemat

def create_datamodule_config(out_path, batch_size, dataset_str):
    datamodule_path = f'{out_path}/configs/datamodule'
    yaml_contents = (
        "_target_: lfads_torch.datamodules.BasicDataModule\n"
        f"datafile_pattern: ${{relpath:datasets/{dataset_str}.h5}}\n"
        f"batch_size: {batch_size}\n"
    )

    with open(f'{datamodule_path}/{dataset_str}.yaml', 'w') as configfile:
        configfile.write(yaml_contents)
    
    print(f'Created datamodule config file for {dataset_str}')

def create_model_config(out_path, dataset_str, binned_trials):
    model_config_dir = f'{out_path}/configs/model'
    template_path = f'{model_config_dir}/toy_data_samp_size_100.yaml'

    with open(template_path, 'r') as template_file:
        model_config = yaml.safe_load(template_file)

    model_config['encod_data_dim'] = binned_trials.shape[2]
    model_config['encod_seq_len'] = binned_trials.shape[1]
    model_config['recon_seq_len'] = binned_trials.shape[1]

    output_path = f'{model_config_dir}/{dataset_str}.yaml'
    with open(output_path, 'w') as output_file:
        yaml.safe_dump(model_config, output_file, sort_keys=False)

    print(f'Created model config file for {dataset_str}')


output_dir = Path(__file__).resolve().parent

if __name__ == '__main__':
    out_path = '/Users/ellamohanram/Documents/GitHub/finding_latent_rates/lfads-torch'
    original_cwd = Path.cwd()
    files_dir = output_dir / "files"
    files_dir.mkdir(parents=True, exist_ok=True)

    DEBUG = False
    # l = 0.01
    l = 0.1 # lambda value for Poisson for the active chanel
    sample_sizes = [100, 120]
    if DEBUG:
        sample_sizes = [10, 12]
    overlaps = [sample_sizes[i]-sample_sizes[0] for i in range(len(sample_sizes))]

    lcm = math.lcm(*sample_sizes) # least common multiple of sample_sizes
    lcm = lcm * 2
    data = np.random.poisson(l, lcm) #number of "spikes" in lcm number of "bins" 
    l = str(l).replace(".", "_")
    if DEBUG:
        data = np.random.randint(3, size=lcm)
    savemat(files_dir / f'toy_data_l{l}.mat', {'data': data})
    print('Saved data in files/')
    print(f'data shape: {data.shape}')
    print(data)
    print('\n')

    split_frac = 0.75 # train - test split
    batch_size = 3 # run LFADS batch

    for i, (sample_size, overlap) in enumerate(zip(sample_sizes, overlaps)):
        print(f'Prepping files for {sample_size} sample size ({overlap} overlap)')
        print('\n')

        #Prep output paths
        train_indices = f"{output_dir}/files/train_indices_toy_ss{sample_size}_l{l}.npy"
        valid_indices = f"{output_dir}/files/valid_indices_toy_ss{sample_size}_l{l}.npy"
        dataset_str = f"toy_data_samp_size_{sample_size}_l{l}"
        data_file = f'{out_path}/datasets/{dataset_str}.h5'

        # Use original data for each iteration (don't mutate it)
        current_data = data.copy()
        
        # Calculate step size: for overlapping windows, step by (sample_size - overlap)
        # For no overlap (overlap=0), step by sample_size
        if overlap == 0:
            step_size = sample_size
        else:
            step_size = sample_size - overlap
            current_data = np.pad(current_data, (overlap, 0), mode='constant', constant_values=0)
        
        # Extract windows
        binned_trials = [current_data[i: i+sample_size] for i in range(0, len(current_data) - sample_size + 1, step_size)]
        binned_trials = np.array(binned_trials)
        binned_trials = binned_trials[:, :, np.newaxis]  
        print(f'binned_trials shape: {binned_trials.shape}')
        np.save(files_dir / f'{dataset_str}_binned_trials.npy', binned_trials)
        print(f'Saved binned trials in {files_dir}')
        if DEBUG:
            print(binned_trials)
        print('\n')

        # Train-val split
        # Randomized train/valid split with index tracking
        n_sessions = binned_trials.shape[0]
        indices = np.arange(n_sessions)

        # Shuffle indices reproducibly
        rng = np.random.default_rng(seed=0)
        rng.shuffle(indices)

        # Compute split point
        split_point = int(n_sessions * split_frac)

        # Split into train and validation indices
        train_idx = indices[:split_point]
        valid_idx = indices[split_point:]

        # Slice data
        train_data = binned_trials[train_idx]
        valid_data = binned_trials[valid_idx]
        print(f"Train shape: {train_data.shape}, Valid shape: {valid_data.shape}")

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
        create_datamodule_config(out_path, batch_size, dataset_str)
        create_model_config(out_path, dataset_str, binned_trials)

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
