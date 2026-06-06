from typing import Annotated

import torch
from pydantic import Field

from gaussian_splatting.structures.renderers.apple_silicon_renderer import (
    AppleSiliconRenderer,
    AppleSiliconRendererParams,
)
from gaussian_splatting.structures.renderers.base_renderer import BaseRenderer
from gaussian_splatting.structures.renderers.naive_renderer import NaiveRenderer, NaiveRendererParams

RendererConfig = Annotated[
    (
        NaiveRendererParams
        | AppleSiliconRendererParams
    ),
    Field(discriminator="name"),
]  # fmt:skip


class RendererFactory:
    REGISTRY = {
        NaiveRendererParams: NaiveRenderer,
        AppleSiliconRendererParams: AppleSiliconRenderer,
    }

    @classmethod
    def create_renderer(
        cls,
        configuration: RendererConfig,
        device: torch.device,
    ) -> BaseRenderer:
        renderer_cls = RendererFactory.REGISTRY[type(configuration)]

        return renderer_cls(
            configuration=configuration,
            device=device,
        )
