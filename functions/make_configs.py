import sys
from pathlib import Path
import yaml
import os

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
    template_path = f'{model_config_dir}/nlb_mc_maze.yaml'

    with open(template_path, 'r') as template_file:
        model_config = yaml.safe_load(template_file)

    model_config['encod_data_dim'] = binned_trials.shape[2]
    model_config['readout']['modules'][0]['out_features'] = binned_trials.shape[2]
    model_config['encod_seq_len'] = binned_trials.shape[1]
    model_config['recon_seq_len'] = binned_trials.shape[1]

    output_path = f'{model_config_dir}/{dataset_str}.yaml'
    with open(output_path, 'w') as output_file:
        yaml.safe_dump(model_config, output_file, sort_keys=False)

    print(f'Created model config file for {dataset_str}')
