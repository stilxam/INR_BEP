from typing import Dict, Any, Optional
from common_dl_utils.config_creation import Config


# def get_config(layer_type: str, activation_kwargs: Dict[str, float]) -> Config:
#     """Create model configuration."""
#     config = Config()
#     config.architecture = "./model_components"
#     config.model_type = "inr_modules.CombinedINR"
#     config.model_config = Config()
#     config.model_config.in_size = 1 # 2
#     config.model_config.out_size = 1 # 2
#     config.model_config.terms = [
#         (
#             "inr_modules.MLPINR.from_config",
#             {
#                 "hidden_size": 1028,
#                 "num_layers": 5,
#                 "layer_type": layer_type,
#                 "num_splits": 1,
#                 "activation_kwargs": activation_kwargs,
#                 "initialization_scheme": "initialization_schemes.siren_scheme",
#                 "initialization_scheme_kwargs": {"w0": 12.0},
#                 "post_processor": "auxiliary.real_scalar",
#             },
#         )
#     ]
#     return config

def get_config(layer_type: str, activation_kwargs: Dict[str, float]) -> Config:
    """Create model configuration."""
    config = Config()
    config.architecture = "./model_components"
    config.model_type = "inr_modules.CombinedINR"
    config.model_config = Config()
    config.model_config.in_size = 1 # 2
    config.model_config.out_size = 1 # 2
    if layer_type == "inr_layers.SIREN":
        init_scheme = "initialization_schemes.siren_scheme"
        init_kwargs = activation_kwargs

    else:
        init_scheme = "initialization_schemes.standard_scheme"
        init_kwargs = {}

    config.model_config.terms = [
        (
            "inr_modules.MLPINR.from_config",
            {
                "hidden_size": 1028,
                "num_layers": 3,
                "layer_type": layer_type,
                "num_splits": 1,
                "activation_kwargs": activation_kwargs,
                "initialization_scheme": init_scheme,
                "initialization_scheme_kwargs": init_kwargs,
                "post_processor": "auxiliary.real_scalar",
            },
        )
    ]


    if layer_type == "inr_layers.FinerLayer":
        config.model_config.terms.append(
            (
                "inr_modules.MLPINR.from_config",
                {
                    "hidden_size": 1028,
                    "num_layers": 3,
                    "layer_type": layer_type,
                    "num_splits": 1,
                    "activation_kwargs": activation_kwargs,
                    "post_processor": "auxiliary.real_scalar",
                },
            )
        )


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
                    "inr_layers.HoscLayer",
                    "inr_layers.SinCardLayer",
                    "inr_layers.GaussianINRLayer",
                    "inr_layers.QuadraticLayer",
                    "inr_layers.MultiQuadraticLayer",
                    "inr_layers.LaplacianLayer",
                    "inr_layers.ExpSinLayer",
                ]
            },
            "w0": {"values":
                       [i for i in range(1, 51, 2)]
                   },
        },
    }



def get_2d_sweep_configuration() -> Dict[str, Any]:
    """Get wandb sweep configuration for two parameters."""
    return {
        "method": "grid",
        "name": "ntk-2d-hyperparameter-sweep",
        "metric": {"name": "ntk_condition_number", "goal": "minimize"},
        "parameters": {
            "layer_type": {
                "values": [
                    "inr_layers.ComplexWIRE",
                    "inr_layers.RealWIRE",
                    "inr_layers.SuperGaussianLayer",
                ]
            },
            "w0": {
                "values":
                    [i for i in range(1, 51, 2)]
            },
            "s0": {
                "values":
                    [i for i in range(1, 51, 2)]
            },
        },
    }

def get_finer_sweep_configuration() -> Dict[str, Any]:
    """Get wandb sweep configuration for two parameters."""
    return {
        "method": "grid",
        "name": "ntk-2d-hyperparameter-sweep",
        "metric": {"name": "ntk_condition_number", "goal": "minimize"},
        "parameters": {
            "layer_type": {
                "values": [
                    "inr_layers.FinerLayer",

                ]
            },
            "w0": {
                "values":
                    [i for i in range(1, 51)]
            },
            "bias_k": {
                "values":
                    [i for i in range(1, 51)]
            },
        },
    }



def get_activation_kwargs(layer_type: str, param1: float, param2: Optional[float] = None) -> dict[str, float]:
    """map layer type to appropriate activation kwargs. handles multiple parameters."""
    if layer_type in [
        "inr_layers.SirenLayer",
        "inr_layers.HoscLayer",
        "inr_layers.SinCardLayer",
    ]:
        return {"w0": param1}  # only one parameter for these layers
    # elif layer_type =="inr_layers.FinerLayer":
    #     return {"w0": param1}
    elif layer_type in [
        "inr_layers.QuadraticLayer",
        "inr_layers.MultiQuadraticLayer",
        "inr_layers.LaplacianLayer",
        "inr_layers.ExpSinLayer"
    ]:
        return {"a": param1}
    elif layer_type in [
        "inr_layers.ComplexWIRE", "inr_layers.RealWIRE"]:
        return {"w0": param1, "s0": param2}
    elif layer_type == "inr_layers.SuperGaussianLayer":
        return {"a": param1, "b": param2}
    elif layer_type == "inr_layers.GaussianINRLayer":
        return {"inverse_scale": param1}
    else:
        raise ValueError(f"unknown layer type: {layer_type}")
