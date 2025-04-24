import os
import yaml
import optuna
import argparse
from pathlib import Path
from typing import List, Dict, Any
import datetime
import numpy as np
from ctf4science.data_module import load_dataset, get_prediction_timesteps, parse_pair_ids, get_applicable_plots
from ctf4science.eval_module import evaluate, save_results
from ctf4science.visualization_module import Visualization
from naive_baselines import NaiveBaseline

file_dir = Path(__file__).parent
results_file = file_dir / 'results.yaml'

def sum_results(results):
    total = 0
    for pair_dict in results['pairs']:
        metric_dict = pair_dict['metrics']
        for metric in metric_dict.keys():
            total += metric_dict[metric]
    return total

def suggest_value(name, param_dict, trial):
    if 'type' not in param_dict:
        raise Exception(f"\'type\' not in {param_dict} keys")
    if 'upper_bound' not in param_dict:
        raise Exception(f"\'upper_bound\' not in {param_dict} keys")
    if 'lower_bound' not in param_dict:
        raise Exception(f"\'lower_bound\' not in {param_dict} keys")
    if 'log' not in param_dict:
        raise Exception(f"\'log\' not in {param_dict} keys")

    if param_dict['type'] == 'float':
        val = trial.suggest_float(name, param_dict['lower_bound'], param_dict['upper_bound'], log= param_dict['log'])
    elif param_dict['type'] == 'int':
        val = trial.suggest_int(name, param_dict['lower_bound'], param_dict['upper_bound'], log= param_dict['log'])
    else:
        raise Exception(f"Invalid parameter type {param_dict['type']}")

    return val

def generate_config(hp_config, template, name, trial):
    # Generate new constant
    val = suggest_value('constant_value', hp_config['hyperparameters']['constant'], trial)
    # Fill out dictionary
    template['model']['constant_value'] = val
    # Save config
    with open(file_dir / 'config' / f'{name}.yaml', 'w') as f:
        yaml.dump(template, f)
    return None

def main(config_path: str) -> None:
    """
    Main function to generate configuration files and run the naive baseline
    model on specified sub-datasets for hyperparameter optimization.

    Loads configuration, generated specific config files, runs model on training
    and validation set, and performs hyperparameter optimization.

    Args:
        config_path (str): Path to the configuration file.
    """
    # Load configuration
    with open(config_path, 'r') as f:
        hp_config = yaml.safe_load(f)

    # Blank dictionary for yaml file
    yaml_dict = {
        'dataset': {
            'name': hp_config['dataset']['name']
        },
        'model': {
            'name': hp_config['model']['name'],
            'method': 'constant',
            'constant_value': None, # This will be replaced
            'train_split': hp_config['model']['train_split'],
            'burn_in_split': hp_config['model']['burn_in_split'],
            'seed': hp_config['model']['seed']
        }
    }

    # Define objective
    def objective(trial):
        # Create config file
        generate_config(hp_config, yaml_dict, 'hp_config', trial)
        # Run model
        command = f'python {file_dir / "run_opt.py"} {file_dir / "config" / "hp_config.yaml"}'
        os.system(command)
        # Extract results
        with open(results_file, 'r') as f:
            results = yaml.safe_load(f)
        score = sum_results(results)
        # Return
        return score

    # Create optuna study with dashboard
    try:
        optuna.delete_study(study_name="ctf-baseline", storage="sqlite:///db.sqlite3")
    except:
        pass
    study = optuna.create_study(
        direction='maximize',
        storage="sqlite:///db.sqlite3",
        study_name="ctf-baseline"
    )
    
    # Run optimization
    study.optimize(objective, n_trials = hp_config['model']['n_trials'])

    # Remove last results file and hp_config.yaml
    results_file.unlink(missing_ok=True)
    (file_dir / 'config' / 'hp_config.yaml').unlink(missing_ok=True)

    # Return prediction matrix using best hyperparameter value
    best_constant = study.best_params['constant_value']
    print(f"Best score: {study.best_value} (params: {study.best_params})")

    # Save final yaml
    yaml_dict['model']['constant_value'] = best_constant
    with open(file_dir / 'config' / f'config_{hp_config["dataset"]["name"]}_constant_batch_all.yaml', 'w') as f:
        yaml.dump(yaml_dict, f)

    # Check dashboard with: `optuna-dashboard sqlite:///db.sqlite3 --port <REMOTE PORT>`
    # Port forward to remote machine with: `ssh -L <LOCAL PORT>:localhost:<REMOTE PORT> <REMOTE>`

    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help="Path to the hyperparameter configuration file.")
    args = parser.parse_args()
    main(args.config)