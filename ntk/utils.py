def layer_name_to_title(layer_name: str) -> str:
    """Convert layer name to display title."""
    titles = {
        "inr_layers.SirenLayer": "SIREN",
        "inr_layers.ComplexWIRE": "Complex WIRE",
        "inr_layers.RealWIRE": "Real WIRE",
        "inr_layers.HoscLayer": "HOSC",
        "inr_layers.SinCardLayer": "Sine Cardinal",
        "inr_layers.GaussianINRLayer": "Gaussian Bump",
        "inr_layers.QuadraticLayer": "Quadratic",
        "inr_layers.MultiQuadraticLayer": "Multi-Quadratic",
        "inr_layers.LaplacianLayer": "Laplacian",
        "inr_layers.SuperGaussianLayer": "Super Gaussian",
        "inr_layers.ExpSinLayer": "Exponential Sine",
    }
    return titles[layer_name]
