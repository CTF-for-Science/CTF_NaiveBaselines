import numpy as np
import importlib
from typing import Optional, Dict
from ctf4science.eval_module import evaluate

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

    def __init__(self, config: Dict, train_data: Optional[np.ndarray] = None, pair_id: Optional[int] = None):
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
        elif self.method == 'optuna':
            self.optuna_params = {
                'lower_bound': config['model']['optuna_lower_bound'],
                'upper_bound': config['model']['optuna_upper_bound'],
                'n_values': config['model']['optuna_n_values'],
                'seed': config['model'].get('random_seed', None)
            }

    def predict(self, test_data: np.ndarray) -> np.ndarray:
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
            pred_data = np.tile(pred_data, (1, test_data.shape[1]))

        elif self.method == 'constant':
            if self.constant_value is None:
                raise ValueError("Constant value is required for 'constant' method.")
            # Predict a constant value across all features and time steps
            pred_data = np.full_like(test_data, self.constant_value)

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
            pred_data = np.full_like(test_data, constants[0])

        elif self.method == 'optuna':
            # Check if optuna is installed
            optuna = import_if_available('optuna')
            if optuna is None:
                raise Exception("Package 'optuna' not installed")
            else:
                print("Imported optuna!")

            # Check inputs
            if self.pair_id is None:
                raise Exception("pair_id is required for `optuna` method.")
            if self.dataset_name is None:
                raise Exception("dataset_name is required for `optuna` method.")

            # Set seed
            seed = self.optuna_params.get('seed', None)
            if seed is not None:
                np.random.seed(seed)

            # Define objective
            def objective(trial):
                # Generate constant
                constant_val = trial.suggest_float('constant_val', self.optuna_params['lower_bound'], self.optuna_params['upper_bound'])
                # Generate prediction
                pred_data = np.full_like(test_data, constant_val)
                # Evaluate prediction
                results = evaluate(self.dataset_name, self.pair_id, test_data, pred_data)
                # For simplicity, optimize 'short_time'
                score = results['short_time'].item()
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
            study.optimize(objective, n_trials = self.optuna_params['n_values'])

            # Return prediction matrix using best hyperparameter value
            best_constant = study.best_params['constant_val']
            pred_data = np.full_like(test_data, best_constant)
            print(f"Best value: {study.best_value} (params: {study.best_params})")

            # Check dashboard with: `optuna-dashboard sqlite:///db.sqlite3 --port <REMOTE PORT>`
            # Port forward to remote machine with: `ssh -L <LOCAL PORT>:localhost:<REMOTE PORT> <REMOTE>`

        else:
            raise ValueError(f"Unknown baseline method: {self.method}")

        return pred_data