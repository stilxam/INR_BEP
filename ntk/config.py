from typing import Dict, Any
from common_dl_utils.config_creation import Config


def get_config(layer_type: str, activation_kwargs: Dict[str, float]) -> Config:
    """Create model configuration."""
    config = Config()
    config.architecture = "./model_components"
    config.model_type = "inr_modules.CombinedINR"
    config.model_config = Config()
    config.model_config.in_size = 2
    config.model_config.out_size = 2
    config.model_config.terms = [
        (
            "inr_modules.MLPINR.from_config",
            {
                "hidden_size": 1028,
                "num_layers": 3,
                "layer_type": layer_type,
                "num_splits": 1,
                "activation_kwargs": activation_kwargs,
                "initialization_scheme": "initialization_schemes.siren_scheme",
                "initialization_scheme_kwargs": {"w0": 12.0},
                "post_processor": "auxiliary.real_scalar",
            },
        )
    ]
    return config

def get_sweep_configuration() -> Dict[str, Any]:
    """Get wandb sweep configuration."""
    return {
        "method": "grid",
        "name": "ntk-layer-sweep",
        "metric": {"name": "ntk_condition_number", "goal": "minimize"},
        "parameters": {
            "layer_type": {
                "values": [
                    "inr_layers.SirenLayer",
                    "inr_layers.ComplexWIRE",
                    "inr_layers.RealWIRE",
                    "inr_layers.HoscLayer",
                    "inr_layers.SinCardLayer",
                    "inr_layers.GaussianINRLayer",
                    "inr_layers.QuadraticLayer",
                    "inr_layers.MultiQuadraticLayer",
                    "inr_layers.LaplacianLayer",
                    "inr_layers.SuperGaussianLayer",
                    "inr_layers.ExpSinLayer",
                ]
            },
            "param_scale": {"values": [0.1, 1.0, 5.0, 10.0, 17.5, 25.0, 32.5, 50]},
        },
    }


def get_activation_kwargs(layer_type: str, param_scale: float) -> Dict[str, float]:
    """Map layer type to appropriate activation kwargs."""
    if layer_type in ["inr_layers.SirenLayer"]:
        return {"w0": param_scale}
    elif layer_type in ["inr_layers.ComplexWIRE", "inr_layers.RealWIRE"]:
        return {"w0": param_scale, "s0": param_scale * 0.6}
    elif layer_type in ["inr_layers.HoscLayer", "inr_layers.SinCardLayer"]:
        return {"w0": param_scale}
    elif layer_type == "inr_layers.GaussianINRLayer":
        return {"inverse_scale": 1.0 / param_scale}
    elif layer_type == "inr_layers.SuperGaussianLayer":
        return {"a": param_scale, "b": param_scale}
    else:
        return {"a": param_scale}

