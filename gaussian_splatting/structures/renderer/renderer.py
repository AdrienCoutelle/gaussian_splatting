from dataclasses import dataclass

import cv2
import mlx.core as mx
import numpy as np
from pydantic import BaseModel, ConfigDict

from gaussian_splatting.structures.camera import Camera
from gaussian_splatting.structures.gaussian import GaussianCollection
from gaussian_splatting.structures.renderer.rasterizer import Rasterizer
from gaussian_splatting.structures.renderer.screen_gaussian import ScreenSpaceGaussians
from gaussian_splatting.structures.renderer.utils import _evaluate_sh, _quaternions_to_rotation_matrices
from gaussian_splatting.utils.logger import Logger
from gaussian_splatting.utils.profiler import profile

logger = Logger("RENDERER")


@dataclass
class Image:
    array: np.ndarray

    @property
    def height(self) -> int:
        return self.array.shape[0]

    @property
    def width(self) -> int:
        return self.array.shape[1]


class RendererConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    width: int
    height: int
    focal_length: float

    gaussian_extent: float = 3.0
    tile_size: int = 16
    max_gaussians_per_batch: int = 1024
    draw_axis: bool = False


@profile
class Renderer:
    def __init__(
        self,
        config: RendererConfig,
    ) -> None:
        self.config = config

        self.rasterizer = Rasterizer(
            gaussian_extent=config.gaussian_extent,
            tile_size=config.tile_size,
            max_gaussians_per_batch=config.max_gaussians_per_batch,
        )

    def render(
        self,
        camera: Camera,
        gaussians: GaussianCollection,
    ) -> Image:
        image_array = np.array(
            self.render_tensor(
                camera=camera,
                gaussians=gaussians,
            )
        )

        if self.config.draw_axis:
            image_array = self._draw_axes(image=image_array, camera=camera)

        return Image(array=image_array)

    def render_tensor(
        self,
        camera: Camera,
        gaussians: GaussianCollection,
    ) -> mx.array:
        gaussians = self._transform_positions_to_camera_space(
            camera=camera,
            gaussians=gaussians,
        )

        screen_space_gaussians = self._project_to_screen_space(
            camera=camera,
            gaussians=gaussians,
        )

        if screen_space_gaussians is None:
            return mx.zeros((camera.h, camera.w, 3), dtype=mx.float32)

        image = self._run_rasterization(
            gaussians=screen_space_gaussians,
            camera=camera,
        )

        return image

    def _transform_positions_to_camera_space(
        self,
        camera: Camera,
        gaussians: GaussianCollection,
    ) -> GaussianCollection:
        r_world_to_camera = camera.pose[:3, :3].T
        camera_center = camera.pose[:3, 3:4]

        positions = (r_world_to_camera @ (gaussians.positions.T - camera_center)).T

        return GaussianCollection.from_tensors(
            positions=positions,
            quaternions=gaussians.quaternions,
            sh_coeffs=gaussians.sh_coeffs,
            scales=gaussians.scales,
            opacities=gaussians.opacities,
        )

    def _project_to_screen_space(
        self,
        camera: Camera,
        gaussians: GaussianCollection,
    ) -> ScreenSpaceGaussians | None:
        principal_point_x, principal_point_y = camera.principal_point

        camera_means = gaussians.positions
        depths = camera_means[:, 2]

        # Cull Gaussians behind the camera (depth ≤ 0 produces invalid projections)
        valid_mask = depths > 0.0
        valid_indices = mx.array(np.where(np.array(valid_mask))[0], dtype=mx.int32)
        if valid_indices.shape[0] == 0:
            return None
        gaussians = gaussians[valid_indices]
        depths = depths[valid_indices]

        means_2d = mx.stack(
            [
                camera.f * (gaussians.positions[:, 0] / depths) + principal_point_x,
                camera.f * (gaussians.positions[:, 1] / depths) + principal_point_y,
            ],
            axis=1,
        )

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
        S_squared = (scales**2)[:, :, None] * mx.eye(3)[None, :, :]
        world_covariances = R @ S_squared @ mx.transpose(R, (0, 2, 1))

        pose = camera.pose
        camera_to_world_rot = pose[:3, :3]
        world_to_camera_rot = camera_to_world_rot.T
        camera_covariances = world_to_camera_rot @ world_covariances @ world_to_camera_rot.T

        covariances_2d = jacobians @ camera_covariances @ mx.transpose(jacobians, (0, 2, 1))

        return ScreenSpaceGaussians(
            means_2d=means_2d,
            covariances_2d=covariances_2d,
            depths=depths,
            colors=self._get_color(
                gaussians=gaussians,
                camera=camera,
            ),
            opacities=1.0 / (1.0 + mx.exp(-gaussians.opacities)),
        )

    def _get_color(
        self,
        gaussians: GaussianCollection,
        camera: Camera,
    ) -> mx.array:
        pose = camera.pose
        camera_to_world_rot = pose[:3, :3]
        norms = mx.clip(mx.sqrt(mx.sum(gaussians.positions**2, axis=1, keepdims=True)), 1e-12, None)
        dirs_camera = -gaussians.positions / norms
        dirs_world = dirs_camera @ camera_to_world_rot.T
        return _evaluate_sh(sh_coeffs=gaussians.sh_coeffs, directions=dirs_world)

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

    def _run_rasterization(
        self,
        gaussians: ScreenSpaceGaussians,
        camera: Camera,
    ) -> None:
        image = self.rasterizer.run(
            gaussians=gaussians,
            camera=camera,
        )

        return mx.clip(image, 0.0, 1.0)
