from typing import Annotated

import torch
from pydantic import Field

from gaussian_splatting.structures.gaussian import Gaussian
from gaussian_splatting.structures.inference_pipelines.base_pipeline import BaseInferencePipeline
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

PipelineConfig = Annotated[
    (
        SingleImageInferencePipelineParams
        | OrbitPipelineInferenceParams
        | InteractiveInferencePipelineParams
    ),
    Field(discriminator="name"),
]  # fmt:skip


class InferencePipelineFactory:
    REGISTRY = {
        SingleImageInferencePipelineParams: SingleImageInferencePipeline,
        OrbitPipelineInferenceParams: OrbitVideoInferencePipeline,
        InteractiveInferencePipelineParams: InteractiveInferencePipeline,
    }

    @staticmethod
    def create(
        renderer: BaseRenderer,
        gaussians: list[Gaussian],
        configuration: PipelineConfig,
        device: torch.device,
        output_folder: str,
        epoch: int | None = None,
    ) -> BaseInferencePipeline:
        pipeline_cls = InferencePipelineFactory.REGISTRY[type(configuration)]

        return pipeline_cls(
            renderer=renderer,
            gaussians=gaussians,
            configuration=configuration,
            output_folder=output_folder,
            device=device,
            epoch=epoch,
        )
