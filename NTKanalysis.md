# Neural Tangent Kernel (NTK) Analysis

This repository provides tools for analyzing Neural Tangent Kernels (NTKs) of various neural network architectures. The primary focus is on computing NTKs, decomposing them, analyzing their frequency spectra, and visualizing the results. The repository also includes configurations for running parameter sweeps using Weights and Biases (wandb).

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Configuration](#configuration)
  - [Running the Sweep](#running-the-sweep)
- [Modules](#modules)
  - [analysis.py](#analysispy)
  - [config.py](#configpy)
  - [data.py](#datapy)
  - [models.py](#modelspy)
  - [sweep.py](#sweeppy)
  - [utils.py](#utilspy)
  - [visualization.py](#visualizationpy)
- [Contributing](#contributing)
- [License](#license)


## Usage

### Configuration

The configuration for the models and sweeps is defined in `config.py`. You can customize the model architecture, layer types, and activation parameters by modifying the configuration functions.

### Running the Sweep

To run the parameter sweep, you need to have Weights and Biases (wandb) installed and configured. You can start the sweep by running:

```bash
python -m ntk.sweep
```

This will initialize wandb, compute the NTKs for different configurations, analyze the results, and log the metrics and visualizations to wandb.

## Modules

### analysis.py

This module provides functions for computing and analyzing NTKs:

- `get_NTK_ntvp(apply_fn: Callable) -> Callable`: Returns a function to compute the NTK.
- `decompose_ntk(ntk: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]`: Decomposes the NTK into eigenvalues and eigenvectors.
- `analyze_fft(ntk: jnp.ndarray) -> jnp.ndarray`: Computes the FFT of the NTK.
- `analyze_fft_spectrum(magnitude_spectrum: jnp.ndarray) -> Dict[str, float]`: Analyzes the FFT magnitude spectrum of the NTK.

### config.py

This module provides functions for creating model configurations and sweep configurations:

- `get_config(layer_type: str, activation_kwargs: Dict[str, float]) -> Config`: Creates a model configuration.
- `get_sweep_configuration() -> Dict[str, Any]`: Returns the wandb sweep configuration.
- `get_activation_kwargs(layer_type: str, param_scale: float) -> Dict[str, float]`: Maps layer types to appropriate activation parameters.

### data.py

This module provides functions for data preparation:

- `get_flattened_locations(n: int) -> jax.Array`: Returns flattened locations for NTK computation.

### models.py

This module provides functions for model initialization and application:

- `make_init_apply(config: Config, key_gen: Generator) -> Tuple[Callable, Callable]`: Creates initialization and apply functions for an MLP model.

### sweep.py

This module provides the main functions for running the parameter sweep:

- `setup_sweep_config() -> tuple[str, float, dict]`: Initializes wandb and gets configuration parameters.
- `compute_ntk(n: int, layer_type: str, activation_kwargs: dict) -> tuple[jnp.ndarray, jnp.ndarray]`: Computes the NTK and its eigenvalues.
- `analyze_and_visualize(NTK: jnp.ndarray, layer_type: str, activation_kwargs: dict) -> tuple[dict, plt.Figure, plt.Figure]`: Analyzes the NTK and creates visualizations.
- `main_sweep() -> None`: Main sweep function.

### utils.py

This module provides utility functions:

- `layer_name_to_title(layer_name: str) -> str`: Converts layer names to display titles.

### visualization.py

This module provides functions for visualizing the results:

- `plot_fft_spectrum(magnitude_spectrum: jnp.ndarray, layer_name: str, activation_kwargs: Dict[str, float]) -> plt.Figure`: Plots the FFT magnitude spectrum.
- `plot_ntk_kernels(NTK: jnp.ndarray, layer_type: str, activation_kwargs: Dict[str, float]) -> plt.Figure`: Plots the NTK kernels.
