from abc import ABC, abstractmethod
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from gaussian_splatting.structures.camera import Camera
from gaussian_splatting.structures.gaussian import Gaussians
from gaussian_splatting.structures.renderer.renderer import Renderer, Visibility

Params = dict[str, mx.array]


class Loss(ABC):
    @abstractmethod
    def compute(
        self,
        image: mx.array,
        gt_image: mx.array,
    ) -> mx.array:
        pass

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


class SSIM:
    """Structural similarity of (H, W, C) images, as in the reference implementation.

    Local statistics use an 11x11 Gaussian window (sigma = 1.5) with zero padding, applied to each channel
    independently; the SSIM map is averaged over pixels and channels.
    """

    C1 = 0.01**2
    C2 = 0.03**2

    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
    ) -> None:
        coords = np.arange(window_size, dtype=np.float32) - window_size // 2
        gauss_1d = np.exp(-(coords**2) / (2.0 * sigma**2))
        gauss_1d /= gauss_1d.sum()
        # conv2d weight layout: (C_out, kH, kW, C_in)
        self.window = mx.array(np.outer(gauss_1d, gauss_1d)[None, :, :, None])
        self.padding = window_size // 2

    def __call__(
        self,
        image: mx.array,
        gt_image: mx.array,
    ) -> mx.array:
        # (H, W, C) -> (C, H, W, 1): each channel is filtered on its own
        x = mx.transpose(image, (2, 0, 1))[..., None]
        y = mx.transpose(gt_image, (2, 0, 1))[..., None]

        mu_x, mu_y = self._filter(x), self._filter(y)
        mu_x_sq, mu_y_sq, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
        sigma_x_sq = self._filter(x * x) - mu_x_sq
        sigma_y_sq = self._filter(y * y) - mu_y_sq
        sigma_xy = self._filter(x * y) - mu_xy

        ssim_map = ((2.0 * mu_xy + self.C1) * (2.0 * sigma_xy + self.C2)) / (
            (mu_x_sq + mu_y_sq + self.C1) * (sigma_x_sq + sigma_y_sq + self.C2)
        )
        return mx.mean(ssim_map)

    def _filter(self, x: mx.array) -> mx.array:
        return mx.conv2d(x, self.window, padding=self.padding)


class L1DSSIMLoss(Loss):
    """(1 - lambda) * L1 + lambda * (1 - SSIM), the paper's loss (Sec. 5.1, lambda = 0.2)."""

    def __init__(
        self,
        lambda_dssim: float = 0.2,
    ) -> None:
        self.lambda_dssim = lambda_dssim
        self.l1 = L1Loss()
        self.ssim = SSIM()

    def compute(
        self,
        image: mx.array,
        gt_image: mx.array,
    ) -> mx.array:
        l1 = self.l1(image, gt_image)
        if self.lambda_dssim == 0.0:
            return l1
        return (1.0 - self.lambda_dssim) * l1 + self.lambda_dssim * (1.0 - self.ssim(image, gt_image))


@dataclass
class RenderingLossOutput:
    loss: float
    grads: Params
    view_space_grads: mx.array  # (N, 2) dL/d(mean_2d) in pixel units; zero for non-visible Gaussians
    visibility: Visibility  # which Gaussians were projected inside the view, and their screen radii


class RenderingLoss:
    def __init__(
        self,
        renderer: Renderer,
        loss: Loss,
    ) -> None:
        self.renderer = renderer
        self.loss = loss
        self._value_and_grad_fn = mx.value_and_grad(self._forward, argnums=(0, 1))

    def evaluate(
        self,
        image: mx.array,
        gt_image: mx.array,
    ) -> float:
        return self.loss(image, gt_image).item()

    def compute_value_and_grad(
        self,
        params: Params,
        camera: Camera,
        gt_image: mx.array,
    ) -> RenderingLossOutput:
        means_2d_offset = mx.zeros((params["positions"].shape[0], 2), dtype=mx.float32)
        visibility: dict[str, Visibility] = {}

        loss, (grads, view_space_grads) = self._value_and_grad_fn(
            params, means_2d_offset, camera, gt_image, visibility
        )
        mx.eval(loss, grads, view_space_grads, visibility["value"].screen_radii)

        return RenderingLossOutput(
            loss=loss.item(),
            grads=grads,
            view_space_grads=view_space_grads,
            visibility=visibility["value"],
        )

    def _forward(
        self,
        params: Params,
        means_2d_offset: mx.array,
        camera: Camera,
        gt_image: mx.array,
        visibility: dict[str, Visibility],
    ) -> mx.array:
        image, visibility["value"] = self.renderer.render_tensor_with_visibility(
            camera=camera,
            gaussians=Gaussians.from_tensors(**params),
            means_2d_offset=means_2d_offset,
        )
        return self.loss(image, gt_image)
