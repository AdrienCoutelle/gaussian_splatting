from dataclasses import dataclass
from enum import StrEnum

import torch

from gaussian_splatting.structures.renderers.apple_silicon_renderer import (
    AppleSiliconRenderer,
    AppleSiliconRendererParams,
)
from gaussian_splatting.structures.renderers.base_renderer import BaseRenderer, RendererParams
from gaussian_splatting.structures.renderers.naive_renderer import NaiveRenderer, NaiveRendererParams


class RendererName(StrEnum):
    NAIVE = "naive"
    APPLE_SILICON = "apple_silicon"
    CUDA = "cuda"


@dataclass
class RendererConfig:
    name: RendererName
    parameters: RendererParams

    @classmethod
    def from_dict(
        cls,
        configuration: dict,
    ) -> "RendererConfig":
        if not isinstance(configuration, dict):
            raise ValueError(f"RendererConfig must be a dictionary, got '{type(configuration).__name__}'.")

        mandatory_fields = {
            "name",
            "parameters",
        }

        if not set(configuration.keys()).issuperset(mandatory_fields):
            missing_fields = mandatory_fields - set(configuration.keys())
            raise ValueError(
                f"RendererConfig is missing the following mandatory fields: {', '.join(missing_fields)}, "
                f"got {', '.join(configuration.keys())}."
            )

        name = configuration["name"]
        if not isinstance(name, str):
            raise TypeError(f"RendererConfig 'name' should be a string, got {type(name).__name__}.")

        if name not in list(RendererName):
            raise ValueError(
                f"RendererConfig supported renderers are {', '.join([s.value for s in RendererName])}, got {name}."
            )

        parameters = configuration["parameters"]

        if name == RendererName.NAIVE:
            return cls(
                name=RendererName.NAIVE,
                parameters=NaiveRendererParams.from_dict(parameters),
            )

        if name == RendererName.APPLE_SILICON:
            return cls(
                name=RendererName.APPLE_SILICON,
                parameters=AppleSiliconRendererParams.from_dict(parameters),
            )

        if name == RendererName.CUDA:
            raise NotImplementedError("CUDA renderer is not implemented yet.")


class RendererFactory:
    @classmethod
    def create_renderer(
        cls,
        configuration: RendererConfig,
        device: torch.device,
    ) -> BaseRenderer:
        renderer_name = configuration.name

        if renderer_name == RendererName.NAIVE:
            assert isinstance(configuration.parameters, NaiveRendererParams)
            return NaiveRenderer(
                configuration=configuration.parameters,
                device=device,
            )

        if renderer_name == RendererName.APPLE_SILICON:
            assert isinstance(configuration.parameters, AppleSiliconRendererParams)
            return AppleSiliconRenderer(
                configuration=configuration.parameters,
                device=device,
            )

        if renderer_name == RendererName.CUDA:
            raise NotImplementedError("CUDA renderer is not implemented yet.")
