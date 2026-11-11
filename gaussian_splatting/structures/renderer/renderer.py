from dataclasses import dataclass
from typing import Annotated

import cv2
import mlx.core as mx
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from gaussian_splatting.structures.camera import Camera
from gaussian_splatting.structures.gaussian import Gaussians
from gaussian_splatting.structures.renderer.rasterizer import Rasterizer, RasterizerConfig
from gaussian_splatting.structures.renderer.screen_gaussian import ScreenSpaceGaussians
from gaussian_splatting.structures.renderer.utils import (
    _batched_matmul,
    _batched_matmul_transposed,
    _evaluate_sh,
    _quaternions_to_rotation_matrices,
)
from gaussian_splatting.utils.logger import Logger
from gaussian_splatting.utils.profiler import profile

logger = Logger("RENDERER")


@dataclass
class Visibility:
    indices: np.ndarray
    screen_radii: mx.array


class RendererConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    draw_axis: bool = False

    near_plane: float = 0.0
    far_plane: float = float("inf")

    background_color: tuple[
        Annotated[float, Field(ge=0.0, le=1.0)],
        Annotated[float, Field(ge=0.0, le=1.0)],
        Annotated[float, Field(ge=0.0, le=1.0)],
    ] = (0.0, 0.0, 0.0)

    rasterizer_config: RasterizerConfig = Field(default_factory=RasterizerConfig)

    @model_validator(mode="after")
    def validate_clipping_planes(self) -> "RendererConfig":
        if self.near_plane > self.far_plane:
            raise ValueError("near_plane must be less than or equal to far_plane")
        return self


@profile
class Renderer:
    def __init__(
        self,
        config: RendererConfig,
    ) -> None:
        self.config = config

        self.rasterizer = Rasterizer(self.config.rasterizer_config)

    def render(
        self,
        camera: Camera,
        gaussians: Gaussians,
    ) -> np.ndarray:
        image_array = np.array(
            self.render_tensor(
                camera=camera,
                gaussians=gaussians,
            )
        )

        if self.config.draw_axis:
            image_array = self._draw_axes(image=image_array, camera=camera)

        return image_array

    def render_tensor(
        self,
        camera: Camera,
        gaussians: Gaussians,
    ) -> mx.array:
        image, _ = self.render_tensor_with_visibility(
            camera=camera,
            gaussians=gaussians,
        )
        return image

    def render_tensor_with_visibility(
        self,
        camera: Camera,
        gaussians: Gaussians,
        means_2d_offset: mx.array | None = None,
    ) -> tuple[mx.array, Visibility]:
        gaussians = self._transform_positions_to_camera_space(
            camera=camera,
            gaussians=gaussians,
        )

        screen_space_gaussians, visible_indices = self._project_to_screen_space(
            camera=camera,
            gaussians=gaussians,
            means_2d_offset=means_2d_offset,
        )

        if screen_space_gaussians is None:
            empty = Visibility(indices=visible_indices, screen_radii=mx.zeros((0,), dtype=mx.float32))
            return self._background_image(camera), empty

        image = self._run_rasterization(
            gaussians=screen_space_gaussians,
            camera=camera,
        )

        screen_radii = mx.stop_gradient(
            screen_space_gaussians.compute_max_extent(extent_factor=self.rasterizer.gaussian_extent)
        )
        return image, Visibility(indices=visible_indices, screen_radii=screen_radii)

    def _transform_positions_to_camera_space(
        self,
        camera: Camera,
        gaussians: Gaussians,
    ) -> Gaussians:
        r_world_to_camera = camera.pose[:3, :3].T
        camera_center = camera.pose[:3, 3:4]

        positions = (r_world_to_camera @ (gaussians.positions.T - camera_center)).T

        return Gaussians.from_tensors(
            positions=positions,
            quaternions=gaussians.quaternions,
            sh_coeffs=gaussians.sh_coeffs,
            scales=gaussians.scales,
            opacities=gaussians.opacities,
        )

    def _project_to_screen_space(
        self,
        camera: Camera,
        gaussians: Gaussians,
        means_2d_offset: mx.array | None = None,
    ) -> tuple[ScreenSpaceGaussians | None, np.ndarray]:
        principal_point_x, principal_point_y = camera.principal_point

        camera_means = gaussians.positions
        depths = camera_means[:, 2]

        valid_mask = (depths >= self.config.near_plane) & (depths <= self.config.far_plane)
        valid_indices_np = np.flatnonzero(np.array(valid_mask))
        if valid_indices_np.shape[0] == 0:
            return None, valid_indices_np
        valid_indices = mx.array(valid_indices_np, dtype=mx.int32)
        gaussians = gaussians[valid_indices]
        depths = depths[valid_indices]

        means_2d = mx.stack(
            [
                camera.f * (gaussians.positions[:, 0] / depths) + principal_point_x,
                camera.f * (gaussians.positions[:, 1] / depths) + principal_point_y,
            ],
            axis=1,
        )

        visible_mask = (
            (means_2d[:, 0] >= 0.0)
            & (means_2d[:, 0] < camera.w)
            & (means_2d[:, 1] >= 0.0)
            & (means_2d[:, 1] < camera.h)
        )
        visible_indices_np = np.flatnonzero(np.array(visible_mask))
        original_indices = valid_indices_np[visible_indices_np]
        if visible_indices_np.shape[0] == 0:
            return None, original_indices
        visible_indices = mx.array(visible_indices_np, dtype=mx.int32)

        gaussians = gaussians[visible_indices]
        depths = depths[visible_indices]
        means_2d = means_2d[visible_indices]
        if means_2d_offset is not None:
            means_2d = means_2d + means_2d_offset[mx.array(original_indices, dtype=mx.int32)]

        zeros = mx.zeros((gaussians.positions.shape[0],))
        row0 = mx.stack(
            [
                camera.f / depths,
                zeros,
                -camera.f * gaussians.positions[:, 0] / (depths**2),
            ],
            axis=1,
        )
        row1 = mx.stack(
            [
                zeros,
                camera.f / depths,
                -camera.f * gaussians.positions[:, 1] / (depths**2),
            ],
            axis=1,
        )
        jacobians = mx.stack([row0, row1], axis=1)

        scales = mx.exp(gaussians.scales)
        R = _quaternions_to_rotation_matrices(gaussians.quaternions)
        RS = R * scales[:, None, :]
        world_covariances = _batched_matmul_transposed(RS, RS)  # (N, 3, 3)

        world_to_camera_rot = camera.pose[:3, :3].T
        T = _batched_matmul(jacobians, mx.broadcast_to(world_to_camera_rot, (jacobians.shape[0], 3, 3)))  # J W
        covariances_2d = _batched_matmul_transposed(_batched_matmul(T, world_covariances), T)  # (N, 2, 2)

        screen_space_gaussians = ScreenSpaceGaussians(
            means_2d=means_2d,
            covariances_2d=covariances_2d,
            depths=depths,
            colors=self._get_color(
                gaussians=gaussians,
                camera=camera,
            ),
            opacities=1.0 / (1.0 + mx.exp(-gaussians.opacities)),
        )
        return screen_space_gaussians, original_indices

    def _get_color(
        self,
        gaussians: Gaussians,
        camera: Camera,
    ) -> mx.array:
        pose = camera.pose
        camera_to_world_rot = pose[:3, :3]
        norms = mx.clip(mx.sqrt(mx.sum(gaussians.positions**2, axis=1, keepdims=True)), 1e-12, None)
        dirs_camera = -gaussians.positions / norms
        dirs_world = dirs_camera @ camera_to_world_rot.T
        return _evaluate_sh(sh_coeffs=gaussians.sh_coeffs, directions=dirs_world)

    def _run_rasterization(
        self,
        gaussians: ScreenSpaceGaussians,
        camera: Camera,
    ) -> mx.array:
        """Rasterizes the Gaussians over the background color.

        Blending is linear in the colors and the blending weights sum to 1 - T_final, so rasterizing the colors
        shifted by -bg and adding bg back gives sum_i w_i c_i + T_final * bg: compositing over the background,
        with exact gradients (including the background's effect on the opacities) and no kernel change.
        """
        background = mx.array(self.config.background_color, dtype=mx.float32)
        shifted_gaussians = ScreenSpaceGaussians(
            means_2d=gaussians.means_2d,
            covariances_2d=gaussians.covariances_2d,
            depths=gaussians.depths,
            colors=gaussians.colors - background,
            opacities=gaussians.opacities,
        )
        image = self.rasterizer.run(
            gaussians=shifted_gaussians,
            camera=camera,
        )

        return mx.clip(image + background, 0.0, 1.0)

    def composite_on_background(
        self,
        image: mx.array,
    ) -> mx.array:
        if image.shape[-1] != 4:
            return image
        background = mx.array(self.config.background_color, dtype=mx.float32)
        rgb, alpha = image[..., :3], image[..., 3:]
        return rgb * alpha + background * (1.0 - alpha)

    def _background_image(
        self,
        camera: Camera,
    ) -> mx.array:
        background = mx.array(self.config.background_color, dtype=mx.float32)
        return mx.broadcast_to(background, (camera.h, camera.w, 3))

    def _draw_axes(
        self,
        image: np.ndarray,
        camera: Camera,
        axis_length: float = 0.5,
        thickness: int = 1,
        opacity: float = 0.3,
    ) -> np.ndarray:
        img_uint8 = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
        axes_overlay = img_bgr.copy()

        origin = np.zeros(3)
        axes = [
            (np.array([axis_length, 0.0, 0.0]), (0, 0, 255)),   # X — red
            (np.array([0.0, axis_length, 0.0]), (0, 255, 0)),   # Y — green
            (np.array([0.0, 0.0, axis_length]), (255, 0, 0)),   # Z — blue
        ]  # fmt: skip

        origin_px = self._project_world_point_to_pixel(point_world=origin, camera=camera)
        if origin_px is None:
            return image

        for axis_end, color in axes:
            end_px = self._project_world_point_to_pixel(point_world=axis_end, camera=camera)
            if end_px is None:
                continue

            direction = np.array(end_px, dtype=np.float64) - np.array(origin_px, dtype=np.float64)
            direction_norm = np.linalg.norm(direction)
            if direction_norm == 0.0:
                continue

            line_extent = np.hypot(image.shape[0], image.shape[1])
            line_direction = direction / direction_norm
            line_start = tuple(np.rint(np.array(origin_px) - line_extent * line_direction).astype(int))
            line_end = tuple(np.rint(np.array(origin_px) + line_extent * line_direction).astype(int))
            cv2.line(axes_overlay, pt1=line_start, pt2=line_end, color=color, thickness=thickness)

        img_bgr = cv2.addWeighted(axes_overlay, opacity, img_bgr, 1.0 - opacity, 0.0)

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_rgb.astype(np.float32) / 255.0

    def _project_world_point_to_pixel(
        self,
        point_world: np.ndarray,
        camera: Camera,
    ) -> tuple[int, int] | None:
        """Project a 3D world-space point to pixel coordinates. Returns None if behind camera."""
        pose = np.array(camera.pose)
        r_world_to_camera = pose[:3, :3].T
        camera_center = pose[:3, 3]
        point_camera = r_world_to_camera @ (point_world - camera_center)

        if point_camera[2] <= 0.0:
            return None

        cx, cy = camera.principal_point
        u = int(camera.f * point_camera[0] / point_camera[2] + cx)
        v = int(camera.f * point_camera[1] / point_camera[2] + cy)
        return (u, v)
