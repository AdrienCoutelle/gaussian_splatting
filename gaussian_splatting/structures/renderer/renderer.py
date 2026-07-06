import math
from dataclasses import dataclass

import mlx.core as mx
import numpy as np
from pydantic import BaseModel, ConfigDict

from gaussian_splatting.structures.camera import Camera
from gaussian_splatting.structures.gaussian import Gaussian, GaussianCollection
from gaussian_splatting.utils.logger import Logger

logger = Logger("RENDERER")

# Spherical harmonics basis coefficients (real, normalized)
_SH_C0 = 0.5 * math.sqrt(1.0 / math.pi)
_SH_C1 = 0.5 * math.sqrt(3.0 / math.pi)
_SH_C2 = [
    0.5 * math.sqrt(15.0 / math.pi),
    -0.5 * math.sqrt(15.0 / math.pi),
    0.25 * math.sqrt(5.0 / math.pi),
    -0.5 * math.sqrt(15.0 / math.pi),
    0.25 * math.sqrt(15.0 / math.pi),
]
_SH_C3 = [
    -0.25 * math.sqrt(35.0 / (2.0 * math.pi)),
    0.5 * math.sqrt(105.0 / math.pi),
    -0.25 * math.sqrt(21.0 / (2.0 * math.pi)),
    0.25 * math.sqrt(7.0 / math.pi),
    -0.25 * math.sqrt(21.0 / (2.0 * math.pi)),
    0.25 * math.sqrt(105.0 / math.pi),
    -0.25 * math.sqrt(35.0 / (2.0 * math.pi)),
]


def _evaluate_sh(
    sh_coeffs: mx.array,
    directions: mx.array,
) -> mx.array:
    """
    Evaluate spherical harmonics at unit viewing directions.

    :param sh_coeffs: SH coefficients of shape (N, num_coeffs, 3), supporting degrees 0–3.
    :param directions: Unit viewing directions of shape (N, 3), from Gaussian toward the camera.
    :return: RGB colors of shape (N, 3), clamped to [0, 1].
    """
    num_coeffs = sh_coeffs.shape[1]

    result = _SH_C0 * sh_coeffs[:, 0, :]  # (N, 3)

    if num_coeffs > 1:
        x = directions[:, 0:1]
        y = directions[:, 1:2]
        z = directions[:, 2:3]
        result = (
            result - _SH_C1 * y * sh_coeffs[:, 1, :] + _SH_C1 * z * sh_coeffs[:, 2, :] - _SH_C1 * x * sh_coeffs[:, 3, :]
        )

    if num_coeffs > 4:
        x = directions[:, 0:1]
        y = directions[:, 1:2]
        z = directions[:, 2:3]
        xx, yy, zz = x * x, y * y, z * z
        xy, yz, xz = x * y, y * z, x * z
        result = (
            result
            + _SH_C2[0] * xy * sh_coeffs[:, 4, :]
            + _SH_C2[1] * yz * sh_coeffs[:, 5, :]
            + _SH_C2[2] * (2.0 * zz - xx - yy) * sh_coeffs[:, 6, :]
            + _SH_C2[3] * xz * sh_coeffs[:, 7, :]
            + _SH_C2[4] * (xx - yy) * sh_coeffs[:, 8, :]
        )

    if num_coeffs > 9:
        x = directions[:, 0:1]
        y = directions[:, 1:2]
        z = directions[:, 2:3]
        xx, yy, zz = x * x, y * y, z * z
        xy = x * y
        result = (
            result
            + _SH_C3[0] * y * (3 * xx - yy) * sh_coeffs[:, 9, :]
            + _SH_C3[1] * xy * z * sh_coeffs[:, 10, :]
            + _SH_C3[2] * y * (4 * zz - xx - yy) * sh_coeffs[:, 11, :]
            + _SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh_coeffs[:, 12, :]
            + _SH_C3[4] * x * (4 * zz - xx - yy) * sh_coeffs[:, 13, :]
            + _SH_C3[5] * z * (xx - yy) * sh_coeffs[:, 14, :]
            + _SH_C3[6] * x * (xx - 3 * yy) * sh_coeffs[:, 15, :]
        )

    return mx.clip(result + 0.5, 0.0, 1.0)


def _quaternions_to_rotation_matrices(quaternions: mx.array) -> mx.array:
    """Convert (N, 4) quaternions [w, x, y, z] to (N, 3, 3) rotation matrices."""
    norms = mx.clip(mx.sqrt(mx.sum(quaternions**2, axis=1, keepdims=True)), 1e-12, None)
    quaternions = quaternions / norms
    w, x, y, z = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]

    return mx.stack(
        [
            1 - 2 * (y**2 + z**2),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x**2 + z**2),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x**2 + y**2),
        ],
        axis=1,
    ).reshape(-1, 3, 3)


@dataclass
class Image:
    array: np.ndarray

    @property
    def height(self) -> int:
        return self.array.shape[0]

    @property
    def width(self) -> int:
        return self.array.shape[1]


@dataclass
class ScreenSpaceGaussians:
    means_2d: mx.array
    covariances_2d: mx.array
    depths: mx.array
    colors: mx.array
    opacities: mx.array


class RendererConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    width: int
    height: int
    focal_length: float

    near_plane: float = 1e-4
    covariance_regularization: float = 0.3
    tile_size: tuple[int, int] = (16, 16)
    max_gaussians_per_tile: int = 4000
    sigma_cut: float = 12.0
    eps: float = 1e-3
    verbose: bool = False


class Renderer:
    def __init__(
        self,
        config: RendererConfig,
    ) -> None:
        self.config = config

    def render(
        self,
        camera: Camera,
        gaussians: list[Gaussian] | GaussianCollection,
    ) -> Image:
        return Image(
            array=np.array(
                self.render_tensor(
                    camera=camera,
                    gaussians=gaussians,
                )
            ),
        )

    def render_tensor(
        self,
        camera: Camera,
        gaussians: list[Gaussian] | GaussianCollection,
    ) -> mx.array:
        """Render to a float array (H, W, 3) in [0, 1]."""
        if isinstance(gaussians, GaussianCollection):  # TODO @Adrien: Remove this check
            gaussian_collection = gaussians
        else:
            gaussian_collection = GaussianCollection(gaussians=gaussians)

        camera_space_gaussians = self._transform_positions_to_camera_space(
            camera=camera,
            gaussians=gaussian_collection,
        )
        logger.info(f"{len(camera_space_gaussians)} gaussians in camera space.")

        screen_space_gaussians = self._project_to_screen_space(
            camera=camera,
            gaussians=camera_space_gaussians,
        )
        logger.info("Gaussians in screen space.")

        output_image = mx.zeros(
            (camera.h, camera.w, 3),
            dtype=mx.float32,
        )

        if screen_space_gaussians is None:
            return output_image

        sorted_indices = mx.argsort(screen_space_gaussians.depths)

        self._splat_gaussians_vectorized(
            image=output_image,
            gaussians=screen_space_gaussians,
            sorted_indices=sorted_indices,
            image_height=camera.h,
            image_width=camera.w,
        )

        return mx.clip(output_image, 0.0, 1.0)

    def _transform_positions_to_camera_space(
        self,
        camera: Camera,
        gaussians: GaussianCollection,
    ) -> GaussianCollection:
        """
        Transform gaussians position and rotation to camera space.

        The pose is the camera-to-world transformation.

        pos_world = r_camera_to_world.pos_camera + t_camera_to_world
        <=> pos_world - t_camera_to_world = r_camera_to_world.pos_camera
        <=> pos_camera = r_world_to_camera.(pos_world - t_camera_to_world)
        """
        r_world_to_camera = camera.pose[:3, :3].T
        camera_center = camera.pose[:3, 3:4]

        camera_means = (r_world_to_camera @ (gaussians.positions.T - camera_center)).T

        return GaussianCollection.from_tensors(
            positions=camera_means,
            quaternions=gaussians.quaternions,
            scales=gaussians.scales,
            sh_coeffs=gaussians.sh_coeffs,
            opacities=gaussians.opacities,
        )

    def _project_to_screen_space(
        self,
        camera: Camera,
        gaussians: GaussianCollection,
    ) -> ScreenSpaceGaussians | None:
        principal_point_x = camera.w / 2.0  # TODO@Adrien: Do this in the Camera class
        principal_point_y = camera.h / 2.0

        # Extract camera means (N, 3)
        camera_means = gaussians.positions
        depths = -camera_means[:, 2]

        # Filter out gaussians behind the near plane (force eval to check count)
        valid_indices = mx.array(np.where(np.array(depths > self.config.near_plane) > 0)[0])
        mx.eval(valid_indices)
        if valid_indices.shape[0] == 0:
            return None

        valid_means = camera_means[valid_indices]
        valid_quaternions = gaussians.quaternions[valid_indices]
        valid_scales = gaussians.scales[valid_indices]
        valid_sh_coeffs = gaussians.sh_coeffs[valid_indices]
        valid_opacities = gaussians.opacities[valid_indices]
        valid_depths = depths[valid_indices]
        N = valid_means.shape[0]

        # Compute world-space viewing directions for SH evaluation.
        # In camera space the camera is at origin, so the direction from Gaussian to camera
        # is -valid_means (camera space). Rotate to world space via the camera-to-world rotation.
        pose = mx.array(camera.pose.detach().cpu().numpy())
        camera_to_world_rot = pose[:3, :3]  # (3, 3)
        norms = mx.clip(mx.sqrt(mx.sum(valid_means**2, axis=1, keepdims=True)), 1e-12, None)
        dirs_camera = -valid_means / norms
        dirs_world = dirs_camera @ camera_to_world_rot.T  # (N, 3)

        valid_colors = _evaluate_sh(sh_coeffs=valid_sh_coeffs, directions=dirs_world)

        # Project to 2D (N, 2)
        means_2d = mx.stack(
            [
                camera.f * (valid_means[:, 0] / valid_depths) + principal_point_x,
                principal_point_y - camera.f * (valid_means[:, 1] / valid_depths),
            ],
            axis=1,
        )

        # Compute Jacobian for all gaussians at once (N, 2, 3)
        zeros = mx.zeros((N,))
        row0 = mx.stack(
            [
                camera.f / valid_depths,
                zeros,
                camera.f * valid_means[:, 0] / (valid_depths**2),
            ],
            axis=1,
        )  # (N, 3)
        row1 = mx.stack(
            [
                zeros,
                -camera.f / valid_depths,
                -camera.f * valid_means[:, 1] / (valid_depths**2),
            ],
            axis=1,
        )  # (N, 3)
        jacobians = mx.stack([row0, row1], axis=1)  # (N, 2, 3)

        # Compute 3D world-space covariance from quaternion and scale: R @ S^2 @ R^T
        R = _quaternions_to_rotation_matrices(valid_quaternions)  # (N, 3, 3)
        S_squared = (valid_scales**2)[:, :, None] * mx.eye(3)[None, :, :]  # (N, 3, 3)
        world_covariances = R @ S_squared @ mx.transpose(R, (0, 2, 1))  # (N, 3, 3)

        # Transform to camera space: W_rot @ Σ_world @ W_rot^T
        world_to_camera_rot = camera_to_world_rot.T  # (3, 3)
        camera_covariances = world_to_camera_rot @ world_covariances @ world_to_camera_rot.T  # (N, 3, 3)

        # Compute 2D covariances: J @ Σ_cam @ J^T for all gaussians
        covariances_2d = jacobians @ camera_covariances @ mx.transpose(jacobians, (0, 2, 1))  # (N, 2, 2)

        # Add regularization
        # regularization = mx.eye(2) * self.config.covariance_regularization
        # covariances_2d += regularization[None]

        return ScreenSpaceGaussians(
            means_2d=means_2d,
            covariances_2d=covariances_2d,
            depths=valid_depths,
            colors=valid_colors,  # RGB after SH evaluation
            opacities=valid_opacities,
        )
