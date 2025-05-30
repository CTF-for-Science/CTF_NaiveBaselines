import numpy as np
import importlib
from typing import Optional, Dict
from ctf4science.eval_module import evaluate_custom

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

class NaiveBaseline:
    """
    A class representing naive baseline models for prediction.

    Handles predictions using 'average', 'constant', or 'random' methods based on the configirl configuration.

    Attributes:
        method (str): The model method ('average', 'constant', or 'random').
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
            self.random_params = {
                'lower_bound': config['model']['random_lower_bound'],
                'upper_bound': config['model']['random_upper_bound'],
                'n_values': config['model']['random_n_values'],
                'distribution': config['model']['random_distribution'],
                'seed': config['model'].get('random_seed', None)
            }

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
            # Generate random constants and pick one arbitrarily (simplified for demo)
            lower_bound = self.random_params['lower_bound']
            upper_bound = self.random_params['upper_bound']
            n_values = self.random_params['n_values']
            distribution = self.random_params['distribution']
            seed = self.random_params.get('seed', None)
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
            pred_data = np.full((self.spatial_dimension, self.prediction_horizon_steps), constants[0])

        else:
            raise ValueError(f"Unknown baseline method: {self.method}")

        return pred_data