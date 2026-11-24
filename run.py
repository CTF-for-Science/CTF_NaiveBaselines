import argparse
import yaml
from pathlib import Path
from typing import List, Dict, Any
import datetime
import numpy as np
from ctf4science.data_module import load_dataset, load_forecast_windows, is_windowed_pair, get_prediction_timesteps, parse_pair_ids, get_applicable_plots
from ctf4science.eval_module import evaluate, save_results
from ctf4science.visualization_module import Visualization
from naive_baselines import NaiveBaseline, WindowedNaiveBaseline, windowed_method_needs_train_data

def main(config_path: str) -> None:
    """
    Main function to run the naive baseline model on specified sub-datasets.

    Loads configuration, parses pair_ids, initializes the model, generates predictions,
    evaluates them, and saves results for each sub-dataset under a batch identifier.

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
    batch_id = f"batch_"
    if config['model']['method'] == 'constant':
        constant_value = config['model']['constant_value']
        batch_id = batch_id + f"{constant_value}"
    elif config['model']['method'] == 'random':
        distribution = config['model']['random_distribution']
        seed = config['model']['random_seed']
        batch_id = batch_id + f"{distribution}_{seed}"
    elif config['model']['method'] == 'average':
        batch_id = batch_id + f"avg"
    elif config['model']['method'] == 'last':
        batch_id = batch_id + "last"
 
    # Define the name of the output folder for your batch
    batch_id = f"{batch_id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

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
        if is_windowed_pair(dataset_name, pair_id):
            # Windowed forecasting pair (e.g. buildings): predict the target channel of every
            # test window, (W, horizon, buildings); evaluate() cuts the truth windows itself
            pred_data = predict_windowed(config, dataset_name, pair_id)
        else:
            pred_data = predict_series(config, dataset_name, pair_id)

        # Evaluate predictions using default metrics
        results = evaluate(dataset_name, pair_id, pred_data)

        # Save results for this sub-dataset and get the path to the results directory
        results_directory = save_results(dataset_name, model_name, batch_id, pair_id, config, pred_data, results)

        # Append metrics to batch results
        # Convert metric values to plain Python floats for YAML serialization
        batch_results['pairs'].append({
            'pair_id': pair_id,
            'metrics': results
        })

        # Generate and save visualizations that are applicable to this dataset
        for plot_type in applicable_plots:
            fig = viz.plot_from_batch(dataset_name, pair_id, results_directory, plot_type=plot_type)
            viz.save_figure_results(fig, dataset_name, model_name, batch_id, pair_id, plot_type, results_directory)

    # Save aggregated batch results
    with open(results_directory.parent / 'batch_results.yaml', 'w') as f:
        yaml.dump(batch_results, f)

def predict_windowed(config: Dict[str, Any], dataset_name: str, pair_id: int) -> np.ndarray:
    """
    Predict every test forecast window of a windowed pair (a pair with a 'horizon', e.g. buildings).

    Args:
        config (Dict[str, Any]): Configuration dictionary.
        dataset_name (str): Name of the dataset.
        pair_id (int): ID of the windowed pair.

    Returns:
        np.ndarray: Predictions of shape (W, horizon, buildings).
    """
    # Lookback windows are a read-only view into the test series; the model reads them in batches
    lookback, covariates = load_forecast_windows(dataset_name, pair_id)
    train_data = load_dataset(dataset_name, pair_id)[0] if windowed_method_needs_train_data(config) else None
    model = WindowedNaiveBaseline(config, pair_id, train_data)
    return model.predict(lookback, covariates)

def predict_series(config: Dict[str, Any], dataset_name: str, pair_id: int) -> np.ndarray:
    """
    Predict the test series of a (non-windowed) pair, in (Timesteps, Features) format.

    Args:
        config (Dict[str, Any]): Configuration dictionary.
        dataset_name (str): Name of the dataset.
        pair_id (int): ID of the pair.

    Returns:
        np.ndarray: Predictions of shape (Timesteps, Features).
    """
    # Load sub-dataset
    train_data, init_data = load_dataset(dataset_name, pair_id, transpose=True)

    # Load initialization matrix if it exists
    if init_data is None:
        # Stack all training matrices to get a single training matrix
        train_data = np.concatenate(train_data, axis=1)
    else:
        # If we are given a burn-in matrix, use it as the training matrix
        train_data = init_data

    # Load metadata (to provide forecast length)
    prediction_timesteps = get_prediction_timesteps(dataset_name, pair_id)
    prediction_horizon_steps = prediction_timesteps.shape[0]

    # Initialize the model with the config and train_data
    model = NaiveBaseline(config, train_data, prediction_horizon_steps, pair_id)

    # Generate predictions
    pred_data = model.predict()
    pred_data = pred_data.T
    return pred_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help="Path to the configuration file.")
    args = parser.parse_args()
    main(args.config)