from abc import ABC, abstractmethod

import mlx.core as mx

from gaussian_splatting.structures.camera import Camera
from gaussian_splatting.structures.gaussian import Gaussians
from gaussian_splatting.structures.renderer.renderer import Renderer

Params = dict[str, mx.array]


class Loss(ABC):
    """Image-space loss comparing a rendered image to its ground truth."""

    @abstractmethod
    def compute(
        self,
        image: mx.array,
        gt_image: mx.array,
    ) -> mx.array:
        """Returns a scalar loss."""

    def __call__(
        self,
        image: mx.array,
        gt_image: mx.array,
    ) -> mx.array:
        return self.compute(image, gt_image)


class L1Loss(Loss):
    def compute(
        self,
        image: mx.array,
        gt_image: mx.array,
    ) -> mx.array:
        return mx.mean(mx.abs(image - gt_image))


class RenderingLoss:
    """Renders Gaussian parameters from a camera and scores the result against a ground truth image.

    Hides the autodiff backend: callers only deal with `evaluate` and `value_and_grad`.
    """

    def __init__(
        self,
        renderer: Renderer,
        loss: Loss,
    ) -> None:
        self.renderer = renderer
        self.loss = loss
        self._value_and_grad_fn = mx.value_and_grad(self._forward)

    def evaluate(
        self,
        image: mx.array,
        gt_image: mx.array,
    ) -> float:
        """Scores an already rendered image (no gradients)."""
        return self.loss(image, gt_image).item()

    def compute_value_and_grad(
        self,
        params: Params,
        camera: Camera,
        gt_image: mx.array,
    ) -> tuple[float, Params]:
        loss, grads = self._value_and_grad_fn(params, camera, gt_image)
        mx.eval(loss, grads)
        return loss.item(), grads

    def _forward(
        self,
        params: Params,
        camera: Camera,
        gt_image: mx.array,
    ) -> mx.array:
        image = self.renderer.render_tensor(
            camera=camera,
            gaussians=Gaussians.from_tensors(**params),
        )
        return self.loss(image, gt_image)
