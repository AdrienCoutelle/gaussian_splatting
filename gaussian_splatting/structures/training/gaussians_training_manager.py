import math
from typing import Annotated

import mlx.core as mx
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from gaussian_splatting.structures.camera import Camera
from gaussian_splatting.structures.gaussian import Gaussians
from gaussian_splatting.structures.renderer.renderer import Visibility
from gaussian_splatting.utils.logger import Logger
from gaussian_splatting.utils.profiler import profile

from .optimizer import GaussianOptimizer, LearningRatesConfig, Params
from .utils import _rotate_by_quaternions

logger = Logger("GAUSSIANS_TRAINING_MANAGER")


class GaussiansTrainingManagerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    learning_rates: LearningRatesConfig

    opacity_reset_interval: Annotated[int, Field(gt=0)] = 3000
    opacity_reset_value: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.01
    prune_opacity_threshold: float = 0.005
    densify_grad_threshold: float = 0.0002
    percent_dense: Annotated[float, Field(gt=0.0)] = 0.01
    prune_world_size_ratio: Annotated[float, Field(gt=0.0)] = 0.1
    max_screen_size: Annotated[float, Field(gt=0.0)] | None = 20.0


@profile
class GaussiansTrainingManager:
    """Owns the trainable Gaussian parameters and everything the training process knows about them.

    This covers the optimizer state, the scene extent, and the per-Gaussian densification statistics
    (accumulated view-space gradients, visibility counts, maximum screen radii). The `Trainer` drives
    the schedule and calls into this class for each step, statistics update, and structural change
    (clone / split / prune / opacity reset).
    """

    def __init__(
        self,
        gaussians: Gaussians,
        cameras: list[Camera],
        configuration: GaussiansTrainingManagerConfig,
    ) -> None:
        self.configuration = configuration

        self.params: Params = {
            "positions": mx.array(gaussians.positions, dtype=mx.float32),
            "quaternions": mx.array(gaussians.quaternions, dtype=mx.float32),
            "scales": mx.array(gaussians.scales, dtype=mx.float32),
            "sh_coeffs": mx.array(gaussians.sh_coeffs, dtype=mx.float32),
            "opacities": mx.array(gaussians.opacities, dtype=mx.float32),
        }

        self.optimizer = GaussianOptimizer(configuration.learning_rates)

        self.scene_extent = self._compute_scene_extent(cameras)
        logger.info(f"Scene extent: {self.scene_extent:.3f}")

        self._reset_densification_stats()

    @property
    def num_gaussians(self) -> int:
        return self.params["positions"].shape[0]

    def to_gaussians(self) -> Gaussians:
        return Gaussians.from_tensors(
            positions=self.params["positions"],
            quaternions=self.params["quaternions"],
            scales=self.params["scales"],
            sh_coeffs=self.params["sh_coeffs"],
            opacities=self.params["opacities"],
        )

    def step(
        self,
        grads: Params,
    ) -> None:
        self.params = self.optimizer.step(self.params, grads)
        mx.eval(self.params)

    def accumulate_densification_stats(
        self,
        view_space_grads: mx.array,
        visibility: Visibility,
        camera: Camera,
    ) -> None:
        visible_indices = visibility.indices
        if visible_indices.shape[0] == 0:
            return

        ndc_scale = mx.array([camera.w * 0.5, camera.h * 0.5], dtype=mx.float32)
        indices = mx.array(visible_indices, dtype=mx.int32)
        grad_norms = mx.linalg.norm(view_space_grads[indices] * ndc_scale, axis=-1)

        self._view_space_grad_accum = self._view_space_grad_accum.at[indices].add(grad_norms)
        self._visibility_count = self._visibility_count.at[indices].add(1.0)
        self._max_screen_radii = self._max_screen_radii.at[indices].maximum(visibility.screen_radii)
        mx.eval(self._view_space_grad_accum, self._visibility_count, self._max_screen_radii)

    def reset_opacities(self) -> None:
        """Caps every opacity at `opacity_reset_value` and clears the matching optimizer state."""
        config = self.configuration
        max_logit = math.log(config.opacity_reset_value / (1.0 - config.opacity_reset_value))
        self.params["opacities"] = mx.minimum(self.params["opacities"], max_logit)
        self.optimizer.reset_state("opacities", num_gaussians=self.num_gaussians)
        mx.eval(self.params)

    def densify_and_prune(
        self,
        iteration: int,
    ) -> dict[str, int]:
        """Clones / splits the under-reconstructed Gaussians, prunes the useless ones, then resets the statistics."""
        config = self.configuration
        num_gaussians = self.num_gaussians

        average_grads = self._view_space_grad_accum / mx.maximum(self._visibility_count, 1.0)
        should_densify = average_grads > config.densify_grad_threshold
        is_large = mx.max(mx.exp(self.params["scales"]), axis=-1) > config.percent_dense * self.scene_extent

        should_split = np.array(should_densify & is_large)
        should_clone = np.array(should_densify & ~is_large)

        keep_idx = np.flatnonzero(~should_split)
        clone_idx = np.flatnonzero(should_clone)
        split_idx = np.flatnonzero(should_split)
        densified_indices = np.concatenate([keep_idx, clone_idx, split_idx, split_idx])
        new_indices = mx.array(densified_indices, dtype=mx.int32)

        new_params = {name: value[new_indices] for name, value in self.params.items()}

        num_split = len(split_idx)
        if num_split > 0:
            split_start = len(keep_idx) + len(clone_idx)
            split_idx = mx.array(split_idx, dtype=mx.int32)
            split_params = self._split_gaussians({name: value[split_idx] for name, value in self.params.items()})
            for name in ("positions", "scales"):
                new_params[name] = mx.concatenate([new_params[name][:split_start], split_params[name]], axis=0)

        num_new = len(densified_indices) - len(keep_idx)
        screen_radii = mx.concatenate(
            [self._max_screen_radii[mx.array(keep_idx, dtype=mx.int32)], mx.zeros((num_new,))]
        )

        prune_masks = self._compute_prune_masks(new_params, screen_radii, iteration)
        should_prune = np.logical_or.reduce(list(prune_masks.values()))

        survivors = np.flatnonzero(~should_prune)
        survivors_mx = mx.array(survivors, dtype=mx.int32)
        self.params = {name: value[survivors_mx] for name, value in new_params.items()}
        self.optimizer.reindex(mx.array(densified_indices[survivors], dtype=mx.int32), num_gaussians=num_gaussians)
        self._reset_densification_stats()
        mx.eval(self.params)

        return {
            "pruned": int(should_prune.sum()),
            **{f"pruned_{reason}": int(mask.sum()) for reason, mask in prune_masks.items()},
            "cloned": len(clone_idx),
            "split": num_split,
            "num_gaussians": self.num_gaussians,
        }

    def _reset_densification_stats(self) -> None:
        self._view_space_grad_accum = mx.zeros((self.num_gaussians,), dtype=mx.float32)
        self._visibility_count = mx.zeros((self.num_gaussians,), dtype=mx.float32)
        self._max_screen_radii = mx.zeros((self.num_gaussians,), dtype=mx.float32)

    @staticmethod
    def _compute_scene_extent(cameras: list[Camera]) -> float:
        """Radius of the camera rig: 1.1 x the largest distance from the mean camera center.

        Follows `getNerfppNorm` in the reference implementation.
        """
        centers = np.stack([np.array(camera.pose[:3, 3]) for camera in cameras])
        distances = np.linalg.norm(centers - centers.mean(axis=0), axis=-1)
        return float(distances.max() * 1.1)

    def _compute_prune_masks(
        self,
        params: Params,
        screen_radii: mx.array,
        iteration: int,
    ) -> dict[str, np.ndarray]:
        """One boolean mask per pruning reason (masks can overlap)."""
        config = self.configuration
        masks = {"low_opacity": np.array(mx.sigmoid(params["opacities"]).reshape(-1) < config.prune_opacity_threshold)}

        # Like the reference, size-based pruning only starts after the first opacity reset
        if iteration > config.opacity_reset_interval:
            max_scales = mx.max(mx.exp(params["scales"]), axis=-1)
            masks["world_size"] = np.array(max_scales > config.prune_world_size_ratio * self.scene_extent)
            if config.max_screen_size is not None:
                masks["screen_size"] = np.array(screen_radii > config.max_screen_size)

        return masks

    @staticmethod
    def _split_gaussians(
        params: Params,
        scale_factor: float = 1.6,
    ) -> Params:
        scales = mx.exp(params["scales"])
        quaternions = params["quaternions"] / mx.linalg.norm(params["quaternions"], axis=-1, keepdims=True)

        offsets = [_rotate_by_quaternions(mx.random.normal(shape=scales.shape) * scales, quaternions) for _ in range(2)]
        positions = mx.concatenate([params["positions"] + offset for offset in offsets], axis=0)

        child_scales = params["scales"] - math.log(scale_factor)
        return {
            "positions": positions,
            "scales": mx.concatenate([child_scales, child_scales], axis=0),
        }
