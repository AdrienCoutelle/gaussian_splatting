# 3D Gaussian Splatting with mlx

This repository implements 3D Gaussian Splatting for novel view synthesis, providing both training and rendering pipelines optimized for Apple Silicon devices.

## Installation

To install in editable mode (also installs dependencies):
````
pip install -e .
````

To install the project with its tests dependencies:
````
pip install -e .[tests]
````

## Launch scripts

You can find configuration file examples [here](config/).

You can use the entrypoints to launch scripts:
````
launch_gaussian_splatting_training /path/to/your/config/file.yaml

launch_gaussian_splatting_inference /path/to/your/config/file.yaml
````

## Reference

This project implements the method described in:

> Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, George Drettakis (2023). *3D Gaussian Splatting for Real-Time Radiance Field Rendering*.  
> https://arxiv.org/abs/2308.04079