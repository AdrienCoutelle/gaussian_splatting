from dataclasses import dataclass

import torch


@dataclass
class Camera:
    pose: torch.Tensor
    focal_length: float
    width: int
    height: int

    @property
    def f(self) -> float:
        return self.focal_length

    @property
    def h(self) -> int:
        return self.height

    @property
    def w(self) -> int:
        return self.width
