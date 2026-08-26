# Inference pipelines

[← Back to README](../README.md)

Several inference pipelines are available to render images and videos from a gaussian splatting scene.

## Launch inference pipelines

Assure the project is installed with its dependencies (see [installation](installation.md)).

Then, you can launch the inference pipelines with the following command:

```bash
python gaussian_splatting/launch_inference.py path/to/your/config.yaml
```

Or use the entrypoint:

```bash
launch_gaussian_splatting_inference path/to/your/config.yaml
```

## Available inference pipelines

See the [configuration templates](config_templates/inference_pipelines/) for the available inference pipelines.

### Common configuration parameters

### Single image inference pipeline

In this pipeline the camera position is chosen by the user. The camera direction (look at point) can be set by the user or automatically computed to look at the mean of the gaussian collection.

### Orbit video inference pipeline

This pipeline generates an orbiting video by moving a camera around the Gaussian scene while varying its distance and elevation, keeping it pointed at a configured point (or mean of the Gaussian collection if not specified).

