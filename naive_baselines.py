import numpy as np
import importlib
from typing import Optional, Dict, List
from ctf4science.data_module import get_config, get_window_params
from ctf4science.eval_module import evaluate_custom

# Number of forecast windows predicted at once by WindowedNaiveBaseline when the config sets no batch_size
DEFAULT_WINDOW_BATCH_SIZE = 4096

def import_if_available(name:str, package:Optional[str]=None):
    """
    A function which attempts to import a module and (optionally) a package from
    that module.

    Attributes:
        name (str): The module to import.
        package (Optional[str]): A package from the module to import.

    Returns:
        The imported package or module.
    """
    try:
        return importlib.import_module(name, package)
    except ImportError:
        return None

def _random_params(config: Dict) -> Dict:
    """Read the parameters of the 'random' method from the model config."""
    return {
        'lower_bound': config['model']['random_lower_bound'],
        'upper_bound': config['model']['random_upper_bound'],
        'n_values': config['model']['random_n_values'],
        'distribution': config['model']['random_distribution'],
        'seed': config['model'].get('random_seed', None)
    }

def _random_constant(random_params: Dict) -> float:
    """
    Draw the constant predicted by the 'random' method.

    Generates `n_values` random constants and returns the first (no evaluation here).

    Raises:
        ValueError: If the distribution is unknown, or a log distribution has a non-positive lower bound.
    """
    lower_bound = random_params['lower_bound']
    upper_bound = random_params['upper_bound']
    n_values = random_params['n_values']
    distribution = random_params['distribution']
    seed = random_params.get('seed', None)
    if seed is not None:
        np.random.seed(seed)

    if distribution == 'uniform':
        constants = np.random.uniform(lower_bound, upper_bound, n_values)
    elif distribution == 'log':
        if lower_bound <= 0:
            raise ValueError("Lower bound must be positive for log distribution")
        constants = np.exp(np.random.uniform(np.log(lower_bound), np.log(upper_bound), n_values))
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    # For simplicity, just use the first constant (no evaluation here)
    return constants[0]

def windowed_method_needs_train_data(config: Dict) -> bool:
    """Return True if WindowedNaiveBaseline needs the training data for the configured method."""
    return config['model']['method'] == 'average'

class NaiveBaseline:
    """
    A class representing naive baseline models for prediction.

    Handles predictions using 'average', 'constant', 'random', or 'last' methods based on the configuration.

    Attributes:
        method (str): The model method ('average', 'constant', 'random', or 'last').
        train_data (Optional[np.ndarray]): Training data, required for 'average' method.
        constant_value (Optional[float]): Constant value for 'constant' method.
        random_params (Optional[Dict]): Parameters for 'random' method.
    """

    def __init__(self, config: Dict, train_data: Optional[np.ndarray] = None, prediction_horizon_steps: int = 0, pair_id: Optional[int] = None):
        """
        Initialize the NaiveBaseline model with the provided configuration.

        Args:
            config (Dict): Configuration dictionary containing method and parameters.
            train_data (Optional[np.ndarray]): Training data for 'average' method.
        """
        self.method = config['model']['method']
        self.dataset_name = config['dataset']['name']
        self.pair_id = pair_id
        self.train_data = train_data
        self.prediction_horizon_steps = prediction_horizon_steps
        self.spatial_dimension = self.train_data.shape[0]

        if self.method == 'constant':
            self.constant_value = config['model']['constant_value']
        elif self.method == 'random':
            self.random_params = _random_params(config)

    def predict(self) -> np.ndarray:
        """
        Generate predictions based on the specified model method.

        Args:
            test_data (np.ndarray): Test data to determine the shape of predictions.

        Returns:
            np.ndarray: Predicted data array.

        Raises:
            ValueError: If the method is unknown or required parameters are missing.
        """
        if self.method == 'average':
            if self.train_data is None:
                raise ValueError("Training data is required for 'average' method.")
            # Predict the mean of each feature from training data, tiled across test time steps
            pred_data = np.mean(self.train_data, axis=1, keepdims=True)
            pred_data = np.tile(pred_data, (1, self.prediction_horizon_steps))

        elif self.method == 'constant':
            if self.constant_value is None:
                raise ValueError("Constant value is required for 'constant' method.")
            # Predict a constant value across all features and time steps
            pred_data = np.full((self.spatial_dimension, self.prediction_horizon_steps), self.constant_value)

        elif self.method == 'random':
            if self.random_params is None:
                raise ValueError("Random search parameters are required for 'random' method.")
            pred_data = np.full((self.spatial_dimension, self.prediction_horizon_steps), _random_constant(self.random_params))

        elif self.method == 'last':
            if self.train_data is None:
                raise ValueError("Training data is required for 'last' method.")
            # Repeat the last observed timestep across the forecast horizon
            pred_data = self.train_data[:, -1:]
            pred_data = np.tile(pred_data, (1, self.prediction_horizon_steps))

        else:
            raise ValueError(f"Unknown baseline method: {self.method}")

        return pred_data


class WindowedNaiveBaseline:
    """
    Naive baselines for windowed forecasting pairs (pairs with a 'horizon' key, e.g. buildings).

    The model predicts the target channel over the horizon of every forecast window, from the
    lookback windows of `ctf4science.data_module.load_forecast_windows` (or the validation loader).
    Windows are predicted in batches of `batch_size` (model config, default
    DEFAULT_WINDOW_BATCH_SIZE), so a large lookback view is never copied in one go.

    Methods:
        last: persistence; repeats the last lookback value of the target channel over the horizon,
            per window and building.
        average: predicts the mean of the target channel over all training data (every train file,
            building and time step), one scalar for every window, step and building.
        constant: predicts `constant_value`.
        random: predicts one constant drawn as in NaiveBaseline.

    Covariates are ignored by every method.

    Attributes:
        method (str): The model method.
        horizon (int): Number of forecast steps per window.
        target_channel (int): Channel of the lookback that is forecast.
        batch_size (int): Number of windows predicted at once.
        value (Optional[float]): The predicted constant, for every method except 'last'.
    """

    def __init__(self, config: Dict, pair_id: int, train_data: Optional[List[np.ndarray]] = None):
        """
        Initialize the windowed baseline for one pair.

        Args:
            config (Dict): Configuration dictionary containing dataset, method and parameters.
            pair_id (int): ID of the windowed pair; sets the horizon and target channel.
            train_data (Optional[List[np.ndarray]]): Training arrays (time, building, channel),
                required for the 'average' method.

        Raises:
            ValueError: If the method is unknown, a parameter is invalid or required data is missing.
        """
        self.method = config['model']['method']
        dataset_name = config['dataset']['name']
        self.horizon = get_window_params(dataset_name, pair_id)['horizon']
        self.target_channel = get_config(dataset_name)['evaluation_params']['target_channel']
        self.batch_size = int(config['model'].get('batch_size', DEFAULT_WINDOW_BATCH_SIZE))
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be a positive integer, got {self.batch_size}")

        self.value = None
        if self.method == 'last':
            pass
        elif self.method == 'average':
            if not train_data:
                raise ValueError("Training data is required for 'average' method.")
            # Pool every train file, building and time step; sum in float64 without copying the arrays
            total = sum(np.sum(t[..., self.target_channel], dtype=np.float64) for t in train_data)
            count = sum(t[..., self.target_channel].size for t in train_data)
            self.value = float(total / count)
        elif self.method == 'constant':
            if config['model'].get('constant_value') is None:
                raise ValueError("Constant value is required for 'constant' method.")
            self.value = float(config['model']['constant_value'])
        elif self.method == 'random':
            self.value = float(_random_constant(_random_params(config)))
        else:
            raise ValueError(f"Unknown baseline method: {self.method}")

    def predict(self, lookback: np.ndarray, covariates: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predict the target channel over the horizon of every window.

        Args:
            lookback (np.ndarray): Lookback windows (W, lookback, buildings, channels); may be a
                read-only view. Only `batch_size` windows are read at a time.
            covariates (Optional[np.ndarray]): Covariate windows (W, horizon, buildings, n_covariates),
                or None. Ignored.

        Returns:
            np.ndarray: Predictions of shape (W, horizon, buildings), in the lookback's float dtype.
        """
        n_windows, _, n_buildings, _ = lookback.shape
        dtype = lookback.dtype if np.issubdtype(lookback.dtype, np.floating) else np.float64
        pred_data = np.empty((n_windows, self.horizon, n_buildings), dtype=dtype)
        for start in range(0, n_windows, self.batch_size):
            stop = min(start + self.batch_size, n_windows)
            if self.method == 'last':
                # Last lookback step of the target channel, (batch, buildings), broadcast over the horizon
                pred_data[start:stop] = lookback[start:stop, -1, :, self.target_channel][:, None, :]
            else:
                pred_data[start:stop] = self.value
        return pred_data
