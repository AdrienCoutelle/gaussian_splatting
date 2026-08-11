# 3D Gaussian Splatting with Apple Silicon

Add simple description and a video example. Explain that it is optimized for apple silicon chips.


## Documentation

- [Project installation](docs/installation.md)
- [Gaussian PLY format](docs/ply-format.md)
- [Dataset preparation with COLMAP](docs/dataset-preparation.md)
- [Inference pipelines](docs/inference.md)
- [Training](docs/training.md)

## References

> Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, George Drettakis (2023).
> *3D Gaussian Splatting for Real-Time Radiance Field Rendering*.
> https://arxiv.org/abs/2308.04079

# TODO:
- The training works but is still a bit slow. I get slower as the number og gaussians grows. Use early stopping when the opacity of the pixel has reached almost 100%? 
- In the rasterizer, try to compile some methods. To do so the inputs have to have the same shape. Use a max_gaussians_per_tile parameter. If less than max_gaussians_per_tile gaussians in the tile, pad with transparent gaussians far away.
- Clean the code. The thing where the size of the gaussians could be handled by another class? It seems it is done twice (or two things are very similar).
- See camera conventions, colmap, gaussian splatting. Make a clear choice, maybe write it in the readme to be sure. The conversion should be done in the colmap wrapper, not in the dataset class.
- Restore some sh for more precision.
- Use metal for parallel tile rasterizing.
- Choose a convention for the camera. If it is not the same as colmap, hanle it directly in colmap, not in the dataset or renderer class.
- Use uv.
- Here I consider all the gaussians for rendering and training. Select only the visible gaussians to be more effective on large scenes.
- Use drawings to explain inference pipelines.