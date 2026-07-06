from dataclasses import dataclass

import mlx.core as mx
import numpy as np
from pydantic import BaseModel, ConfigDict

from gaussian_splatting.structures.camera import Camera
from gaussian_splatting.structures.gaussian import GaussianCollection
from gaussian_splatting.structures.renderer.rasterizer import Rasterizer
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


@dataclass
class ScreenSpaceGaussians:
    means_2d: mx.array
    covariances_2d: mx.array
    depths: mx.array
    colors: mx.array
    opacities: mx.array

    def __len__(self) -> int:
        return int(self.means_2d.shape[0])


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


@profile
class Renderer:
    def __init__(
        self,
        config: RendererConfig,
    ) -> None:
        self.config = config

    def render(
        self,
        camera: Camera,
        gaussians: GaussianCollection,
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
        gaussians: GaussianCollection,
    ) -> mx.array:
        self._transform_positions_to_camera_space(
            camera=camera,
            gaussians=gaussians,
        )

        screen_space_gaussians = self._project_to_screen_space(
            camera=camera,
            gaussians=gaussians,
        )

        if screen_space_gaussians is None:
            return mx.zeros((camera.h, camera.w, 3), dtype=mx.float32)

        sorted_indices = mx.argsort(screen_space_gaussians.depths)

        output_image = self._splat_gaussians(
            gaussians=screen_space_gaussians,
            sorted_indices=sorted_indices,
            camera=camera,
        )

        return mx.clip(output_image, 0.0, 1.0)

    def _transform_positions_to_camera_space(
        self,
        camera: Camera,
        gaussians: GaussianCollection,
    ) -> None:
        """
        Transform gaussians position and rotation to camera space.

        The pose is the camera-to-world transformation.

        pos_world = r_camera_to_world.pos_camera + t_camera_to_world
        <=> pos_world - t_camera_to_world = r_camera_to_world.pos_camera
        <=> pos_camera = r_world_to_camera.(pos_world - t_camera_to_world)
        """
        r_world_to_camera = camera.pose[:3, :3].T
        camera_center = camera.pose[:3, 3:4]

        gaussians.positions = (r_world_to_camera @ (gaussians.positions.T - camera_center)).T

    def _project_to_screen_space(
        self,
        camera: Camera,
        gaussians: GaussianCollection,
    ) -> ScreenSpaceGaussians | None:
        """
        To project points:
        [u,v,1]^T = K@[x/z, y/z, 1]^T
        where K is the camera intrinsics matrix:
        K = [
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1]
        ]
        u = f * x/z + cx
        v = f * y/z + cy
        """
        principal_point_x, principal_point_y = camera.principal_point

        camera_means = gaussians.positions
        depths = -camera_means[:, 2]

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

        means_2d = mx.stack(
            [
                camera.f * (valid_means[:, 0] / valid_depths) + principal_point_x,
                -camera.f * (valid_means[:, 1] / valid_depths) + principal_point_y,  # flip y-axis for image coordinates
            ],
            axis=1,
        )

        pose = camera.pose
        camera_to_world_rot = pose[:3, :3]  # (3, 3)
        norms = mx.clip(mx.sqrt(mx.sum(valid_means**2, axis=1, keepdims=True)), 1e-12, None)
        dirs_camera = -valid_means / norms
        dirs_world = dirs_camera @ camera_to_world_rot.T  # (N, 3)

        valid_colors = _evaluate_sh(sh_coeffs=valid_sh_coeffs, directions=dirs_world)

        zeros = mx.zeros((valid_means.shape[0],))
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

        R = _quaternions_to_rotation_matrices(valid_quaternions)  # (N, 3, 3)
        S_squared = (valid_scales**2)[:, :, None] * mx.eye(3)[None, :, :]  # (N, 3, 3)
        world_covariances = R @ S_squared @ mx.transpose(R, (0, 2, 1))  # (N, 3, 3)

        world_to_camera_rot = camera_to_world_rot.T  # (3, 3)
        camera_covariances = world_to_camera_rot @ world_covariances @ world_to_camera_rot.T  # (N, 3, 3)

        covariances_2d = jacobians @ camera_covariances @ mx.transpose(jacobians, (0, 2, 1))  # (N, 2, 2)

        return ScreenSpaceGaussians(
            means_2d=means_2d,
            covariances_2d=covariances_2d,
            depths=valid_depths,
            colors=valid_colors,
            opacities=valid_opacities,
        )

    def _splat_gaussians(
        self,
        gaussians: ScreenSpaceGaussians,
        sorted_indices: mx.array,
        camera: Camera,
    ) -> mx.array:
        if len(gaussians) == 0:
            return mx.zeros((camera.h, camera.w, 3), dtype=mx.float32)

        means_2d = gaussians.means_2d[sorted_indices]  # (N, 2)
        covariances_2d = gaussians.covariances_2d[sorted_indices]  # (N, 2, 2)
        colors = gaussians.colors[sorted_indices]  # (N, 3)
        opacities = gaussians.opacities[sorted_indices]  # (N,) or (N, 1)

        # Compute conics: inverse of 2D covariance [[a, b], [b, c]]
        # stored as (q11, q12, q22) = (c/det, -b/det, a/det)
        a = covariances_2d[:, 0, 0]
        b = covariances_2d[:, 0, 1]
        c = covariances_2d[:, 1, 1]
        det = mx.maximum(a * c - b * b, 1e-10)
        inv_det = 1.0 / det
        conic = mx.stack([c * inv_det, -b * inv_det, a * inv_det], axis=1)  # (N, 3)

        if opacities.ndim > 1:
            opacities = opacities.squeeze(-1)

        tx, ty = self.config.tile_size
        tile_width = (camera.w + tx - 1) // tx
        tile_height = (camera.h + ty - 1) // ty
        num_tiles = tile_width * tile_height

        # Tile binning in numpy: assign each Gaussian to a tile by its projected mean.
        # Stable sort preserves the depth order within each tile.
        xys_np = np.array(means_2d)
        tile_x = (xys_np[:, 0] / tx).astype(np.int32)
        tile_y = (xys_np[:, 1] / ty).astype(np.int32)
        in_bounds = (tile_x >= 0) & (tile_x < tile_width) & (tile_y >= 0) & (tile_y < tile_height)
        valid_idx = np.where(in_bounds)[0]

        tile_gstart_np = np.zeros(num_tiles, dtype=np.uint32)
        tile_gcount_np = np.zeros(num_tiles, dtype=np.uint32)
        reordered_ids = np.empty(0, dtype=np.int32)

        if len(valid_idx) > 0:
            valid_tile_ids = tile_y[valid_idx] * tile_width + tile_x[valid_idx]
            order = np.argsort(valid_tile_ids, kind="stable")
            reordered_ids = valid_idx[order].astype(np.int32)
            sorted_tile_ids = valid_tile_ids[order]

            unique_tiles, first_occ, counts = np.unique(
                sorted_tile_ids,
                return_index=True,
                return_counts=True,
            )
            for tile_id, start, count in zip(unique_tiles, first_occ, counts):
                tile_gstart_np[tile_id] = start
                tile_gcount_np[tile_id] = min(count, self.config.max_gaussians_per_tile)

        if len(reordered_ids) == 0:
            return mx.zeros((camera.h, camera.w, 3), dtype=mx.float32)

        tile_indices = np.arange(num_tiles, dtype=np.uint32)
        tile_origins_np = np.stack(
            [
                (tile_indices % tile_width) * tx,
                (tile_indices // tile_width) * ty,
            ],
            axis=1,
        ).astype(np.uint32)

        ids_mx = mx.array(reordered_ids, dtype=mx.int32)
        gauss_xy = mx.take(means_2d, ids_mx, axis=0).astype(mx.float32)
        gauss_conic = mx.take(conic, ids_mx, axis=0).astype(mx.float32)
        gauss_opacity = mx.take(opacities, ids_mx, axis=0).astype(mx.float32)
        gauss_color = mx.take(colors, ids_mx, axis=0).astype(mx.float32)

        return Rasterizer().rasterize(
            gauss_xy=gauss_xy,
            gauss_conic=gauss_conic,
            gauss_opacity=gauss_opacity,
            gauss_color=gauss_color,
            tile_origins=mx.array(tile_origins_np, dtype=mx.uint32),
            tile_gstart=mx.array(tile_gstart_np, dtype=mx.uint32),
            tile_gcount=mx.array(tile_gcount_np, dtype=mx.uint32),
            image_width=camera.w,
            image_height=camera.h,
            tile_size=self.config.tile_size,
            sigma_cut=self.config.sigma_cut,
            eps=self.config.eps,
        )
