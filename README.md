# INR_BEP: Modular Framework for Training and Benchmarking Implicit Neural Representations

This repository offers a flexible and extensible framework for research and education on Implicit Neural Representations (INRs). It supports a range of architectures, layer types, activation functions, positional encodings, and data modalities.

---

## Repository Structure

- **inr_utils/**  
  Utilities for training single INRs (coordinate MLPs/neural fields): includes training loops, loss functions, metrics, image utilities, callbacks, and more.

- **hypernetwork_utils/**  
  Tools for training hypernetworks, with a similar setup to `inr_utils`.

- **model_components/**  
  Core modules for INRs and hypernetworks:
  - `inr_layers.py`: Various INR layer types & positional encoding layers
  - `activation_functions.py`: Supported activation functions
  - `inr_modules.py`: High-level model modules (MLPs, NeRF, etc.)
  - `hypernetwork_components.py`: Hypernetwork-specific logic
  - Support modules: `auxiliary.py`, `initialization_schemes.py`, `nerf_components.py`

- **Example Notebooks**  
  Demonstrations for images, audio, and more.

---

## INR Layer Types

A wide variety of INR layers are implemented, including:

- Sinusoidal Representation Networks (SIREN)
- Cardinal Sine (sinc)
- Higher-order sine/cosine (Hosc, AdaHosc)
- Wavelet-inspired (Real/Complex WIRE)
- Linear (no activation)
- Gaussian-featured, Quadratic, Multi-Quadratic, Laplacian, Super-Gaussian, Exponential-Sine, and FINER layers

All layers are compatible with hypernetwork-based weight generation.

---

## Activation Functions

Choose from a modular set of activations, such as:

- Sinusoidal (SIREN)
- Sinc / Cardinal Sine
- HOSC and Adaptive HOSC
- Quadratic & Multi-Quadratic
- Laplacian & Super-Gaussian
- Exponential Sine
- FINER (variable-periodic sine)
- Unscaled Gaussian Bump, Real/Complex Gabor Wavelets
- Linear (identity)

Activation functions can be selected or customized in your configuration.

---

## Positional Encodings

Available as plug-and-play layers:

- Classical (NeRF-style Fourier features)
- Trident (from the TRIDENT paper)
- Integer Lattice (learnable, stateful, with dynamic pruning)

---

## Supported Modalities

- **Images:** Continuous representations, super-resolution, inpainting, and more
- **Audio:** 1D signal modeling (see `inr_audio_ex.ipynb`)
- **3D/Volumes:** Neural/radiance fields (NeRF)
- **General:** Any coordinate-to-value function

---

## Example: Model Configuration

Define architectures in a Python config dictionary:

```python
config = dict(
    architecture="my_model_code.py",
    model_type="AutoEncoder",
    encoder="ConvEncoder",
    encoder_config=dict(hidden_channels=64, kernel_size=3, depth=6),
    auxiliary_regressor="Regressor",
    auxiliary_regressor_config=dict(pred_size=10, depth=4),
    decoder="INRDecoder",
    decoder_config=dict(
        mlp_depth=3,
        mlp_width=1024,
        inr_depth=6,
        inr_layer_type="inr_layers.SirenLayer",
        inr_layer_kwargs={"w0": 30.}
    ),
    latent_size=64,
    data_channels=3,
    ndim=2,
)
```

---

## Neural Tangent Kernels (NTKs) & Spectral Bias

This repository includes tools for analyzing spectral bias in INRs using NTKs. Spectral bias describes a neural network’s preference for learning low-frequency components before high-frequency ones. By computing the NTK spectrum, you can quantitatively compare architectures, layer types, and activation functions.

- **Modular NTK computation:** Analyze any INR or hypernetwork model.
- **Frequency response:** Probe how model design affects learning across frequencies.
- **Example notebooks:** Demonstrations for reproducible NTK analysis.

---

## Extensibility

- Add new layers, activations, or encodings by extending `model_components`.
- Swap encoders, decoders, or positional encodings by editing configuration.
- Designed for clarity, collaboration, and robust experimentation.

---

For further details, consult the module docstrings and example notebooks.

Questions and contributions are welcome!

---

