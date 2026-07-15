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


# TODO:
- The training works but is still a bit slow. I get slower as the number og gaussians grows. Use early stopping when the opacity of the pixel has reached almost 100%? 
- Add number of gaussians + time of rendering in the tensorboard logs. Also add images (gt and rendered image side to side.) rather than saving images on disk. As for nerf, select a random image to do the validation pose. Maybe not fully random to be able to compare iterations.
- Recreate the dataset with 128x128 or 256x256 images.
- In the rasterizer, try to compile some methods. To do so the inputs have to have the same shape. Use a max_gaussians_per_tile parameter. If less than max_gaussians_per_tile gaussians in the tile, pad with transparent gaussians far away.
- Clean the code. The thing where the size of the gaussians could be handled by another class? It seems it is done twice (or two things are very similar).
- See camera conventions, colmap, gaussian splatting. Make a clear choice, maybe write it in the readme to be sure. The conversion should be done in the colmap wrapper, not in the dataset class.
- Restore some sh for more precision.