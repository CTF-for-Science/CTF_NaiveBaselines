# CTF_NaiveBaselines

This directory contains naive baseline methods for the CTF for Science framework. These baselines serve as simple benchmarks for comparison with more sophisticated models.

## Baselines

- **Average**: Predicts the average of the training data for each feature, repeated across all test time steps.
- **Constant**: Predicts a specified constant value for all features and time steps. To use the zero baseline, set `constant_value: 0.0`.
- **Random**: Generates random predictions based on a specified distribution (e.g., uniform or normal).

## Usage

To run a baseline, use the `run.py` script from the **project root** followed by the path to a configuration file. For example:

```bash
python models/CTF_NaiveBaselines/run.py models/CTF_NaiveBaselines/config/config_Lorenz_average_batch_1-6.yaml
```

**Important**: Always run the script from the project root to ensure relative paths (e.g., to datasets and results directories) work correctly.

This command executes the average baseline on the specified dataset (e.g., `ODE_Lorenz`) for the sub-datasets defined in the config file (e.g., sub-datasets 1 through 6). Results, including predictions, evaluation metrics, and visualizations, are saved in `results/<dataset_name>/CTF_NaiveBaselines_<method>/<batch_id>/`.

### Batch Run Approach

The `run.py` script supports batch runs across multiple sub-datasets as specified in the configuration file. It:
- Parses the `pair_id` from the config to determine which sub-datasets to process.
- Generates a unique `batch_id` for the run (e.g., `batch__20250404_164642`).
- Processes each sub-dataset, saving results and visualizations in a structured directory.
- Aggregates metrics in a `batch_results.yaml` file.

## Configuration Files

Configuration files are located in the `models/CTF_NaiveBaselines/config/` directory and specify the dataset, sub-datasets, and baseline method, along with method-specific parameters.

### Available Configuration Files

- `config_Lorenz_average_batch_1-6.yaml`: Runs the average baseline on `ODE_Lorenz` for sub-datasets 1 through 6.
- `config_Lorenz_constant_batch_123456.yaml`: Runs the constant baseline on `ODE_Lorenz` for sub-datasets 1, 2, 3, 4, 5, and 6.
- `config_Lorenz_random_batch_1-6.yaml`: Runs the random baseline on `ODE_Lorenz` for sub-datasets 1 through 6.
- `config_KS_average_batch_all.yaml`: Runs the average baseline on `PDE_KS` for all sub-datasets.
- `config_KS_constant_batch_all.yaml`: Runs the constant baseline on `PDE_KS` for all sub-datasets.
- `config_KS_random_batch_all.yaml`: Runs the random baseline on `PDE_KS` for all sub-datasets.

### Configuration Structure

Each configuration file must include:
- **`dataset`** (required):
  - `name`: The dataset name (e.g., `ODE_Lorenz`, `PDE_KS`).
  - `pair_id`: Specifies sub-datasets to run on. Formats:
    - Single integer: `pair_id: 3`
    - List: `pair_id: [1, 2, 3, 4, 5, 6]`
    - Range string: `pair_id: '1-6'`
    - Omitted or `'all'`: Runs on all sub-datasets.
- **`model`**:
  - `name`: Typically `CTF_NaiveBaselines`.
  - `method`: The baseline method (`average`, `constant`, `random`).
  - Method-specific parameters:
    - For `constant`: `constant_value` (e.g., `0.0` for zero baseline).
    - For `random`: `random_distribution` (e.g., `uniform`, `normal`), `random_seed`, etc.

Example (`models/CTF_NaiveBaselines/config/config_Lorenz_average_batch_1-6.yaml`):
```yaml
dataset:
  name: ODE_Lorenz
  pair_id: '1-6'  # Runs on sub-datasets 1 through 6
model:
  name: CTF_NaiveBaselines
  method: average
```

### 🔢 Available Methods

| Method     | Description                                      | Config Example                          |
|------------|--------------------------------------------------|------------------------------------------|
| `average`  | Predicts the mean of the training data           | `method: average`                       |
| `constant` | Predicts a specified constant value              | `method: constant`<br>`constant_value: 0.0` |
| `random`   | Generates random predictions from a distribution | `method: random`<br>`random_distribution: uniform` |

## Examples

- **Average Baseline on Lorenz sub-datasets 1-6**:
  ```bash
  python models/CTF_NaiveBaselines/run.py models/CTF_NaiveBaselines/config/config_Lorenz_average_batch_1-6.yaml
  ```

- **Constant Baseline (zero baseline) on KS all sub-datasets**:
  ```bash
  python models/CTF_NaiveBaselines/run.py models/CTF_NaiveBaselines/config/config_KS_constant_batch_all.yaml
  ```

- **Random Baseline on Lorenz sub-datasets 1-6**:
  ```bash
  python models/CTF_NaiveBaselines/run.py models/CTF_NaiveBaselines/config/config_Lorenz_random_batch_1-6.yaml
  ```

## Requirements

The baselines rely on packages already in the main `requirements.txt`:
- numpy
- scipy
- pyyaml
- matplotlib (for visualizations)

No additional dependencies are required.

## Notes

- The `random` baseline generates predictions based on a specified distribution (e.g., uniform or normal), differing from a random search over constants as in earlier implementations.
- Ensure configuration files match the desired dataset and sub-datasets.
- Results are saved with a unique `batch_id` to prevent overwriting and organize runs.
- Visualizations are automatically generated and saved in `results/<dataset>/<model>/<batch_id>/<pair_id>/visualizations/`.