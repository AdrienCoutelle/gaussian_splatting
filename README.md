# 3D Gaussian Splatting

## Installation

To install in editable mode (also installs dependencies):
````
pip install -e .
````

To install the project with its tests dependencies:
````
pip install -e .[tests]
````

# TODO

- Look at the regularization thing. What is it ? How to setup ? Looks like it changed the aspect of the fly...
- Clean the inference code.
- Write a Apple Silicon compatible version of the renderer.
- Write a CUDA compatible version of the renderer.
- Make the whole code faster. Delete de Gaussian class, only use the Gaussians class (instead of GaussianCollection)
- Clean the torch type issue so I don't have to convert everything to float32 in the renderer code.
- Write the COLMAP wrapper
- Write the training loop