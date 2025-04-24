import argparse
import shutil
import yaml
from pathlib import Path
from typing import List, Dict, Any
import datetime
import numpy as np
from ctf4science.data_module import load_dataset, get_prediction_timesteps, parse_pair_ids, get_applicable_plots
from ctf4science.eval_module import evaluate_custom, save_results
from ctf4science.visualization_module import Visualization
from naive_baselines import NaiveBaseline

# Delete results directory - used for storing batch_results
file_dir = Path(__file__).parent
# TODO

# Notes:
# K value larger than 10 results in invalid spatio-temporal loss
# Currently just overwriting config file and results file to save space

def load_validation_dataset(dataset_name, pair_id, train_split=0.8, burn_in_split=0.4):
    # Load sub-dataset
    train_data, init_data = load_dataset(dataset_name, pair_id)

    # Stack all training matrices to get a single training matrix
    train_data_all = np.concatenate(train_data, axis=1)

    # Calculate total number of training points
    train_num = int(train_split*train_data_all.shape[1])

    # Generate validation split
    if pair_id == 8:
        # Validation split is obtained from the matrices X5train and X6train
        # Calculate number of points of first two train matrices to use for the training split
        train_split_num = train_num - train_data[2].shape[1]
        if train_split_num <= 0:
            raise Exception(f"train_split of {train_split} is too small. Not enough data for (pair_id = {pair_id}) test.")
        # Extract training split from first two train matrices
        train_data[0] = np.concatenate([train_data[0], train_data[1]], axis=1)
        train_data_tmp = train_data[0].copy()
        train_data[0] = train_data_tmp[:, 0:train_split_num]
        train_data.pop(1)
        # Extract validation split
        val_data = train_data_tmp[:, train_split_num:]
    elif pair_id == 9:
        # Validation split is obtained from matrix X8train
        # Calculate number of points of third train matrix to use for the training split
        train_split_num = train_num - train_data[0].shape[1] - train_data[1].shape[1]
        if train_split_num <= 0:
            raise Exception(f"train_split of {train_split} is too small. Not enough data for pair_id = {pair_id} test.")
        # Extract training split from third train matrix
        train_data_tmp = train_data[2].copy()
        train_data[2] = train_data_tmp[:, 0:train_split_num]
        # Extract validation split
        val_data = train_data_tmp[:, train_split_num:]
    else:
        # Validation split is obtained from only matrix in the train_data list
        train_data_tmp = train_data[0].copy()
        train_data[0] = train_data_tmp[:, 0:train_num]
        val_data = train_data_tmp[:, train_num:]

    # Extract burn in split when applicable
    if pair_id in [8, 9]:
        val_split_num = int(burn_in_split*val_data.shape[1])
        val_data_original = val_data.copy()
        val_data = val_data_original[:, 0:val_split_num]
        init_data = val_data_original[:, val_split_num:]

    return train_data, val_data, init_data

def main(config_path: str) -> None:
    """
    Main function to run the naive baseline model on specified sub-datasets.

    Loads configuration, parses pair_ids, initializes the model, generates predictions,
    evaluates them, and saves results for each sub-dataset under a batch identifier.

    The evaluation function evaluates on validation data obtained from training data.

    Args:
        config_path (str): Path to the configuration file.
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load dataset name and get list of sub-dataset train/test pair_ids
    dataset_name = config['dataset']['name']
    pair_ids = parse_pair_ids(config['dataset'])

    model_name = f"{config['model']['name']}_{config['model']['method']}"

    # Generate a unique batch_id for this run, you can add any descriptions you want
    #   e.g. f"batch_{learning_rate}_"
    batch_id = "hyper_opt"
 
    # Initialize batch results dictionary for summary
    batch_results = {
        'batch_id': batch_id,
        'model': model_name,
        'dataset': dataset_name,
        'pairs': []
    }

    # Initialize Visualization object
    viz = Visualization()

    # Get applicable visualizations for the dataset
    applicable_plots = get_applicable_plots(dataset_name)

    # Process each sub-dataset
    for pair_id in pair_ids:
        # Generate training and validation splits (and burn-in matrix when applicable) 
        train_split = config['model']['train_split']
        burn_in_split = config['model']['burn_in_split']
        train_data, val_data, init_data = load_validation_dataset(dataset_name, pair_id, train_split, burn_in_split)

        # Load initialization matrix if it exists
        if init_data is None:
            # Stack all training matrices to get a single training matrix
            train_data = np.concatenate(train_data, axis=1)
        else:
            # If we are given a burn-in matrix, use it as the training matrix
            train_data = init_data

        # Load metadata (to provide forecast length)
        prediction_timesteps = None # TODO
        prediction_horizon_steps = val_data.shape[1]

        # Initialize the model with the config and train_data
        model = NaiveBaseline(config, train_data, prediction_horizon_steps, pair_id)

        # Generate predictions
        pred_data = model.predict()

        # Evaluate predictions using default metrics
        results = evaluate_custom(dataset_name, pair_id, val_data, pred_data)

        # Save results for this sub-dataset and get the path to the results directory
        results_directory = save_results(dataset_name, model_name, batch_id, pair_id, config, pred_data, results)

        # Append metrics to batch results
        # Convert metric values to plain Python floats for YAML serialization
        batch_results['pairs'].append({
            'pair_id': pair_id,
            'metrics': results
        })

    # Save aggregated batch results
    with open(results_directory.parent / 'batch_results.yaml', 'w') as f:
        yaml.dump(batch_results, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help="Path to the configuration file.")
    args = parser.parse_args()
    main(args.config)