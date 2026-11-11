import datetime
import math
import os
from typing import Annotated

import cv2
import mlx.core as mx
import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from gaussian_splatting.structures.camera import Camera
from gaussian_splatting.structures.dataset import GaussianSplattingDataset
from gaussian_splatting.structures.gaussian import Gaussians
from gaussian_splatting.structures.renderer.renderer import Renderer, Visibility
from gaussian_splatting.utils.differentiability_check import check_renderer_differentiability
from gaussian_splatting.utils.image import stack_images_horizontally
from gaussian_splatting.utils.logger import Logger
from gaussian_splatting.utils.ply.ply_saver import PLYSaver
from gaussian_splatting.utils.profiler import profile

from .loss import L1DSSIMLoss, RenderingLoss
from .optimizer import GaussianOptimizer, LearningRatesConfig
from .utils import _rotate_by_quaternions

logger = Logger("TRAINER")


class TrainerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    iterations: Annotated[int, Field(gt=0)]
    learning_rates: LearningRatesConfig
    save_every_n_iterations: Annotated[int, Field(gt=0)] = 5000
    render_scale: Annotated[float, Field(gt=0.0, le=1.0)] = 1.0
    gradient_accumulation_steps: Annotated[int, Field(gt=0)] = 1
    max_gaussians_per_step: int = 0  # 0 = no limit; positive = random subsample per step
    log_every_n_iterations: Annotated[int, Field(gt=0)] = 100
    validate_every_n_iterations: Annotated[int, Field(gt=0)] = 500

    lambda_dssim: Annotated[float, Field(ge=0.0, le=1.0)] = 0.2

    densification_interval: Annotated[int, Field(gt=0)] = 100
    densify_from_iteration: Annotated[int, Field(ge=0)] = 500
    densify_until_iteration: Annotated[int, Field(ge=0)] = 15000  # no densification nor opacity reset from here on
    opacity_reset_interval: Annotated[int, Field(gt=0)] = 3000
    opacity_reset_value: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.01
    prune_opacity_threshold: float = 0.005
    densify_grad_threshold: float = 0.0002
    percent_dense: Annotated[float, Field(gt=0.0)] = 0.01
    prune_world_size_ratio: Annotated[float, Field(gt=0.0)] = 0.1
    max_screen_size: Annotated[float, Field(gt=0.0)] | None = 20.0


@profile
class Trainer:
    def __init__(
        self,
        gaussians: Gaussians,
        renderer: Renderer,
        dataset: GaussianSplattingDataset,
        output_folder: str,
        configuration: TrainerConfig,
    ) -> None:
        self.renderer = renderer
        self.rendering_loss = RenderingLoss(
            renderer=renderer,
            loss=L1DSSIMLoss(lambda_dssim=configuration.lambda_dssim),
        )
        self.dataset = dataset
        self.output_folder = output_folder
        self.configuration = configuration

        check_renderer_differentiability(self.renderer)

        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(os.path.join(output_folder, "checkpoints"))
        self.validation_images_folder = os.path.join(output_folder, "validation")
        os.makedirs(self.validation_images_folder)

        tb_log_dir = os.path.join(output_folder, "tensorboard")
        self.tensorboard_writer = SummaryWriter(log_dir=tb_log_dir)
        logger.info(f"TensorBoard logs: {tb_log_dir}")

        self.params = {
            "positions": mx.array(gaussians.positions, dtype=mx.float32),
            "quaternions": mx.array(gaussians.quaternions, dtype=mx.float32),
            "scales": mx.array(gaussians.scales, dtype=mx.float32),
            "sh_coeffs": mx.array(gaussians.sh_coeffs, dtype=mx.float32),
            "opacities": mx.array(gaussians.opacities, dtype=mx.float32),
        }

        self.optimizer = GaussianOptimizer(configuration.learning_rates)

        self.scene_extent = self._compute_scene_extent(dataset.cameras)
        logger.info(f"Scene extent: {self.scene_extent:.3f}")

        self._reset_densification_stats()

    def run(self) -> None:
        config = self.configuration
        logger.info(f"Starting training for {config.iterations} iterations over {len(self.dataset)} cameras.")

        running_loss = 0.0
        self._run_validation(iteration=0)

        for iteration in tqdm(range(1, config.iterations + 1), desc="Training"):
            running_loss += self._run_iteration(iteration)

            if iteration % config.log_every_n_iterations == 0:
                self.tensorboard_writer.add_scalar(
                    tag="Loss/train",
                    scalar_value=running_loss / config.log_every_n_iterations,
                    global_step=iteration,
                )
                self.tensorboard_writer.add_scalar(
                    tag="Stats/num_gaussians",
                    scalar_value=self._num_gaussians,
                    global_step=iteration,
                )
                running_loss = 0.0

            if iteration % config.validate_every_n_iterations == 0:
                self._run_validation(iteration)

            if iteration % config.save_every_n_iterations == 0:
                self._save_checkpoint(iteration)

        self.tensorboard_writer.close()

    @property
    def _num_gaussians(self) -> int:
        return self.params["positions"].shape[0]

    def _run_iteration(
        self,
        iteration: int,
    ) -> float:
        camera_index = mx.random.randint(0, len(self.dataset)).item()
        gt_image, camera = self.dataset[camera_index]

        output = self.rendering_loss.compute_value_and_grad(
            params=self.params,
            camera=camera,
            gt_image=gt_image,
        )
        self.params = self.optimizer.step(self.params, output.grads)
        mx.eval(self.params)

        config = self.configuration
        if iteration < config.densify_until_iteration:
            self._accumulate_densification_stats(output.view_space_grads, output.visibility, camera)
            mx.eval(self._view_space_grad_accum, self._visibility_count, self._max_screen_radii)

            if iteration > config.densify_from_iteration and iteration % config.densification_interval == 0:
                stats = self._run_adaptive_density_control(iteration)
                for name, count in stats.items():
                    self.tensorboard_writer.add_scalar(f"Densification/{name}", count, iteration)

            if iteration % config.opacity_reset_interval == 0:
                self._reset_opacities()

        return output.loss

    def _reset_opacities(self) -> None:
        max_logit = math.log(self.configuration.opacity_reset_value / (1.0 - self.configuration.opacity_reset_value))
        self.params["opacities"] = mx.minimum(self.params["opacities"], max_logit)
        self.optimizer.reset_state("opacities", num_gaussians=self._num_gaussians)
        mx.eval(self.params)

    def _reset_densification_stats(self) -> None:
        self._view_space_grad_accum = mx.zeros((self._num_gaussians,), dtype=mx.float32)
        self._visibility_count = mx.zeros((self._num_gaussians,), dtype=mx.float32)
        self._max_screen_radii = mx.zeros((self._num_gaussians,), dtype=mx.float32)

    @staticmethod
    def _compute_scene_extent(cameras: list[Camera]) -> float:
        """Radius of the camera rig: 1.1 x the largest distance from the mean camera center (reference `getNerfppNorm`)."""
        centers = np.stack([np.array(camera.pose[:3, 3]) for camera in cameras])
        distances = np.linalg.norm(centers - centers.mean(axis=0), axis=-1)
        return float(distances.max() * 1.1)

    def _accumulate_densification_stats(
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

    def _run_adaptive_density_control(
        self,
        iteration: int,
    ) -> dict[str, int]:
        config = self.configuration
        num_gaussians = self.params["positions"].shape[0]

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
            "num_gaussians": self._num_gaussians,
        }

    def _compute_prune_masks(
        self,
        params: dict[str, mx.array],
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
        params: dict[str, mx.array],
        scale_factor: float = 1.6,
    ) -> dict[str, mx.array]:
        scales = mx.exp(params["scales"])
        quaternions = params["quaternions"] / mx.linalg.norm(params["quaternions"], axis=-1, keepdims=True)

        offsets = [_rotate_by_quaternions(mx.random.normal(shape=scales.shape) * scales, quaternions) for _ in range(2)]
        positions = mx.concatenate([params["positions"] + offset for offset in offsets], axis=0)

        child_scales = params["scales"] - math.log(scale_factor)
        return {
            "positions": positions,
            "scales": mx.concatenate([child_scales, child_scales], axis=0),
        }

    def _save_checkpoint(
        self,
        iteration: int | None = None,
    ) -> None:
        gaussians = Gaussians.from_tensors(
            positions=self.params["positions"],
            quaternions=self.params["quaternions"],
            scales=self.params["scales"],
            sh_coeffs=self.params["sh_coeffs"],
            opacities=self.params["opacities"],
        )

        file_name = (
            f"checkpoint_iteration_{iteration}"
            if iteration is not None
            else "checkpoint_final"
        )  # fmt:skip

        ply_saver = PLYSaver(os.path.join(self.output_folder, "checkpoints", f"{file_name}.ply"))
        ply_saver.save_gaussians(gaussians)

    def _run_validation(
        self,
        iteration: int,
    ) -> None:
        gt_image, camera = self.dataset.validation_item
        gt_image = self.renderer.composite_on_background(gt_image)

        gaussians = Gaussians.from_tensors(
            positions=self.params["positions"],
            quaternions=self.params["quaternions"],
            scales=self.params["scales"],
            sh_coeffs=self.params["sh_coeffs"],
            opacities=self.params["opacities"],
        )

        t0 = datetime.datetime.now()
        image = self.renderer.render_tensor(
            camera=camera,
            gaussians=gaussians,
        )
        mx.eval(image)
        render_time = (datetime.datetime.now() - t0).total_seconds()

        val_loss = self.rendering_loss.evaluate(image, gt_image)
        mse = mx.mean((mx.clip(image, 0.0, 1.0) - gt_image) ** 2).item()
        psnr = -10.0 * math.log10(max(mse, 1e-10))

        # Ground truth on the left, render on the right
        stacked_validation_image = np.clip(
            stack_images_horizontally(left_image=np.array(gt_image), right_image=np.array(image)), 0.0, 1.0
        )

        self.tensorboard_writer.add_scalar("Loss/validation", val_loss, iteration)
        self.tensorboard_writer.add_scalar("Metrics/validation_psnr", psnr, iteration)
        self.tensorboard_writer.add_scalar("Stats/num_gaussians", self._num_gaussians, iteration)
        self.tensorboard_writer.add_scalar("Stats/render_time", render_time, iteration)
        self.tensorboard_writer.add_image("Validation/Image", stacked_validation_image, iteration, dataformats="HWC")
        self.tensorboard_writer.flush()

        cv2.imwrite(
            os.path.join(self.validation_images_folder, f"iteration_{iteration:06d}.png"),
            cv2.cvtColor((stacked_validation_image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
        )
