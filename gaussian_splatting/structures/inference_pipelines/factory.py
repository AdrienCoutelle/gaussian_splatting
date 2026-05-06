from dataclasses import dataclass
from enum import StrEnum

import torch

from gaussian_splatting.structures.gaussian import Gaussian
from gaussian_splatting.structures.inference_pipelines.base_pipeline import (
    BaseInferencePipeline,
    InferencePipelineParams,
)
from gaussian_splatting.structures.inference_pipelines.interactive_pipeline import (
    InteractiveInferencePipeline,
    InteractiveInferencePipelineParams,
)
from gaussian_splatting.structures.inference_pipelines.orbit_video_inference_pipeline import (
    OrbitPipelineInferenceParams,
    OrbitVideoInferencePipeline,
)
from gaussian_splatting.structures.inference_pipelines.single_image_pipeline import (
    SingleImageInferencePipeline,
    SingleImageInferencePipelineParams,
)
from gaussian_splatting.structures.renderers.base_renderer import BaseRenderer


class InferencePipelineName(StrEnum):
    POSITION = "position"
    ORBIT = "orbit"
    INTERACTIVE = "interactive"


@dataclass
class InferencePipelineConfig:
    name: InferencePipelineName
    parameters: InferencePipelineParams

    @staticmethod
    def from_dict(configuration: dict) -> "InferencePipelineConfig":
        if not isinstance(configuration, dict):
            raise ValueError(f"InferencePipelineConfig must be a dictionary, got '{type(configuration).__name__}'.")

        mandatory_fields = {
            "name",
            "parameters",
        }

        if not set(configuration.keys()).issuperset(mandatory_fields):
            missing_fields = mandatory_fields - set(configuration.keys())
            raise ValueError(
                f"InferencePipelineConfig is missing the following mandatory fields: {', '.join(missing_fields)}, "
                f"got {', '.join(configuration.keys())}."
            )

        name = configuration["name"]
        if not isinstance(name, str):
            raise TypeError(f"InferencePipelineConfig 'name' should be a string, got {type(name).__name__}.")

        if name not in list(InferencePipelineName):
            raise ValueError(
                f"InferencePipelineConfig supported pipelines are "
                f"{', '.join([s.value for s in InferencePipelineName])}, got {name}."
            )

        parameters = configuration["parameters"]

        if name == InferencePipelineName.POSITION:
            return InferencePipelineConfig(
                name=name,
                parameters=SingleImageInferencePipelineParams.from_dict(parameters),
            )

        if name == InferencePipelineName.ORBIT:
            return InferencePipelineConfig(
                name=name,
                parameters=OrbitPipelineInferenceParams.from_dict(parameters),
            )

        if name == InferencePipelineName.INTERACTIVE:
            return InferencePipelineConfig(
                name=name,
                parameters=InteractiveInferencePipelineParams.from_dict(parameters),
            )


class InferencePipelineFactory:
    @staticmethod
    def create(
        renderer: BaseRenderer,
        gaussians: list[Gaussian],
        configuration: InferencePipelineConfig,
        device: torch.device,
        output_folder: str,
        epoch: int | None = None,
    ) -> BaseInferencePipeline:
        pipeline_name = configuration.name

        if pipeline_name == InferencePipelineName.POSITION:
            assert isinstance(configuration.parameters, SingleImageInferencePipelineParams)
            return SingleImageInferencePipeline(
                renderer=renderer,
                gaussians=gaussians,
                configuration=configuration.parameters,
                device=device,
                output_folder=output_folder,
                epoch=epoch,
            )

        if pipeline_name == InferencePipelineName.ORBIT:
            assert isinstance(configuration.parameters, OrbitPipelineInferenceParams)
            return OrbitVideoInferencePipeline(
                renderer=renderer,
                gaussians=gaussians,
                configuration=configuration.parameters,
                output_folder=output_folder,
                device=device,
                epoch=epoch,
            )

        if pipeline_name == InferencePipelineName.INTERACTIVE:
            assert isinstance(configuration.parameters, InteractiveInferencePipelineParams)
            return InteractiveInferencePipeline(
                renderer=renderer,
                gaussians=gaussians,
                configuration=configuration.parameters,
                output_folder=output_folder,
                device=device,
                epoch=epoch,
            )

        raise ValueError(f"Unsupported pipeline type: {pipeline_name}")
