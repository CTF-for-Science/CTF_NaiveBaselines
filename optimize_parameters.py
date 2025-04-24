import os
import sys
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

# Update python PATH so that we can load run.py from CTF_NaiveBaselines directly
sys.path.insert(0, str(file_dir))

from run_opt import main as run_opt_main

def sum_results(results):
    """
    Sums all metric values from a results dictionary containing evaluation metrics.
    
    Iterates through all pairs in the results dictionary and sums all metric values
    found in each pair's 'metrics' dictionary. This is used to aggregate
    evaluation metrics from a batch_results.yaml file.

    Args:
        results (dict): A dictionary containing evaluation results.
    
    Returns:
        float: The sum of all metric values across all pairs in the results dictionary.
    """
    total = 0
    for pair_dict in results['pairs']:
        metric_dict = pair_dict['metrics']
        for metric in metric_dict.keys():
            total += metric_dict[metric]
    return total

def suggest_value(name, param_dict, trial):
    """
    Suggest a value for a hyperparameter using Optuna's trial object.

    This function provides a unified interface for suggesting both float and integer
    parameter values within specified bounds, with optional logarithmic scaling.

    Args:
        name (str): The name of the hyperparameter to optimize. This will be used as the
                    identifier in the Optuna study.
        param_dict (dict):
            Dictionary containing the parameter specification with the following keys:
            - 'type': str, either 'float' or 'int' indicating the parameter type
            - 'lower_bound': float/int, the minimum value for the parameter
            - 'upper_bound': float/int, the maximum value for the parameter
            - 'log': bool, whether to sample in log space
        trial (optuna.trial.Trial):
            The Optuna trial object used for suggesting parameter values.

    Returns:
        float or int: The suggested value for the parameter, type depends on param_dict['type']

    Raises:
        Exception:
            If any of the required keys ('type', 'lower_bound', 'upper_bound', 'log')
            are missing from param_dict.
            If the parameter type is neither 'float' nor 'int'.

    Examples:

    >>> param_dict = {
    ...     'type': 'float',
    ...     'lower_bound': 1e-5,
    ...     'upper_bound': 1e-2,
    ...     'log': True
    ... }
    >>> value = suggest_value('learning_rate', param_dict, trial)
    """
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
    """
    Generates a configuration file with suggested hyperparameter values.

    This function suggests a value for the constant parameter using Optuna's trial,
    updates the configuration template with this value, and saves the resulting
    configuration to a YAML file.

    Args:
        hp_config (dict): Dictionary containing a hyperparameter information.
        template (dict): Configuration template dictionary that will be populated with
            the suggested values.
        name (str): Name to use for the output configuration file (without extension).
        trial (optuna.Trial): Optuna trial object used for suggesting parameter values.

    Returns:
        dict: updated config file

    Side Effects:
        - Writes a new YAML configuration file to ./config/{name}.yaml
        - Modifies the input template dictionary by adding the suggested constant value
    """
    # Generate new constant
    val = suggest_value('constant_value', hp_config['hyperparameters']['constant'], trial)
    # Fill out dictionary
    template['model']['constant_value'] = val
    # Save config
    config_path = file_dir / 'config' / f'{name}.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(template, f)
    return config_path

def main(config_path: str, save_config: bool = True) -> None:
    """
    Main function to generate configuration files and run the naive baseline
    model on specified sub-datasets for hyperparameter optimization.

    Loads configuration, generates specific config files, runs model on training
    and validation set, and performs hyperparameter optimization.

    Args:
        config_path (str): Path to the configuration file.
        save_config (str): Save the final configuration file. (only False in unit tests)
    """
    # Load configuration
    with open(config_path, 'r') as f:
        hp_config = yaml.safe_load(f)

    # Blank dictionary for runnable yaml file
    yaml_dict = {
        'dataset': {
            'name': hp_config['dataset']['name']
        },
        'model': {
            'name': hp_config['model']['name'],
            'method': 'constant',
            'constant_value': None, # This will be replaced
            'train_split': hp_config['model']['train_split'],
            'seed': hp_config['model']['seed']
        }
    }

    # Define objective for Optuna
    def objective(trial):
        # Create config file
        config_path = generate_config(hp_config, yaml_dict, 'hp_config', trial)
        # Run model
        run_opt_main(config_path)
        # Extract results
        with open(results_file, 'r') as f:
            results = yaml.safe_load(f)
        score = sum_results(results)
        # Return score
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

    # Remove last results file and hp_config.yaml (no loose files)
    results_file.unlink(missing_ok=True)
    (file_dir / 'config' / 'hp_config.yaml').unlink(missing_ok=True)

    # Obtain best hyperparameter value
    best_constant = study.best_params['constant_value']
    print(f"Best score: {study.best_value} (params: {study.best_params})")

    # Save final configuration yaml from hyperparameter optimization
    if not save_config: # Only False when unit testing
        print("Not saving final config file.")
    else:
        config_path = file_dir / 'config' / f'config_{hp_config["dataset"]["name"]}_constant_batch_all.yaml'
        yaml_dict['model']['constant_value'] = best_constant
        print("Final config file saved to:", config_path)
        with open(config_path, 'w') as f:
            yaml.dump(yaml_dict, f)

    # You can check the Optuna dashboard with: `optuna-dashboard sqlite:///db.sqlite3 --port <REMOTE PORT>`
    # Port forward to a remote machine with: `ssh -L <LOCAL PORT>:localhost:<REMOTE PORT> <REMOTE>`

    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help="Path to the hyperparameter configuration file.")
    parser.add_argument('save_config', action='store_true', help="Save the final hyperparameter configuration file. Only used when unit testing.")
    args = parser.parse_args()
    main(args.config, args.save_config)