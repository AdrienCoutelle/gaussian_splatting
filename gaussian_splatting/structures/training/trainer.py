import os
from typing import Annotated

import cv2
import mlx.core as mx
import mlx.optimizers as opt
import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from tqdm import tqdm

from gaussian_splatting.structures.dataset import GaussianSplattingDataset
from gaussian_splatting.structures.gaussian import GaussianCollection
from gaussian_splatting.structures.renderer.renderer import Renderer
from gaussian_splatting.structures.training.utils import ssim
from gaussian_splatting.utils.differentiability_check import check_renderer_differentiability
from gaussian_splatting.utils.logger import Logger
from gaussian_splatting.utils.profiler import profile
from gaussian_splatting.utils.tensorboard import TensorBoardWriter

logger = Logger("TRAINER")


class TrainerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    epochs: Annotated[int, Field(gt=0)]
    learning_rate: float = 1e-3
    save_every_n_epochs: int = 10
    render_scale: Annotated[float, Field(gt=0.0, le=1.0)] = 1.0
    gradient_accumulation_steps: Annotated[int, Field(gt=0)] = 1
    max_gaussians_per_step: int = 0  # 0 = no limit; positive = random subsample per step
    log_every_n_epochs: int = 5  # log metrics + image to TensorBoard every N epochs

    # Simplified Densification Settings
    densification_interval: int = 10  # Run densification every N steps
    prune_opacity_threshold: float = 0.005
    densify_grad_threshold: float = 0.0002
    split_scale_threshold: float = 0.01


@profile
class Trainer:
    def __init__(
        self,
        gaussians_collection: GaussianCollection,
        renderer: Renderer,
        dataset: GaussianSplattingDataset,
        output_folder: str,
        configuration: TrainerConfig,
    ) -> None:
        self.renderer = renderer
        self.dataset = dataset
        self.output_folder = output_folder
        self.configuration = configuration

        check_renderer_differentiability(self.renderer)

        os.makedirs(output_folder, exist_ok=True)

        tb_log_dir = os.path.join(output_folder, "tensorboard")
        self.tensorboard_writer = TensorBoardWriter(tb_log_dir)
        logger.info(f"TensorBoard logs: {tb_log_dir}")

        self.params = {
            "positions": mx.array(gaussians_collection.positions, dtype=mx.float32),
            "quaternions": mx.array(gaussians_collection.quaternions, dtype=mx.float32),
            "scales": mx.array(gaussians_collection.scales, dtype=mx.float32),
            "sh_coeffs": mx.array(gaussians_collection.sh_coeffs, dtype=mx.float32),
            "opacities": mx.array(gaussians_collection.opacities, dtype=mx.float32),
        }

        self.optimizer = opt.Adam(learning_rate=configuration.learning_rate)

        # Gradient and step accumulation variables
        num_gaussians = self.params["positions"].shape[0]
        self.pos_grad_accum = mx.zeros((num_gaussians,), dtype=mx.float32)
        self.denom = mx.zeros((num_gaussians,), dtype=mx.float32)
        self.step_count = 0

    def run(self) -> None:
        logger.info(f"Starting training for {self.configuration.epochs} epochs over {len(self.dataset)} cameras.")

        for epoch in tqdm(range(1, self.configuration.epochs + 1), desc="Training"):
            avg_loss = self._run_epoch(epoch)

            logger.info(f"Epoch {epoch}/{self.configuration.epochs} completed — Avg Loss: {avg_loss:.6f}")
            self.tensorboard_writer.log_scalar("Loss/train_epoch", avg_loss, epoch)

        self.tensorboard_writer.close()

    def _run_epoch(
        self,
        epoch: int,
    ) -> float:
        indices = mx.random.permutation(len(self.dataset)).tolist()

        epoch_loss = 0.0
        num_rendered = 0

        def loss_fn(params, camera, gt_image):
            gaussians = self._build_gaussian_collection(params)
            image = self.renderer.render_tensor(
                camera=camera,
                gaussians=gaussians,
            )
            cv2.imwrite(
                "image.jpg",
                (np.array(image) * 255).astype("uint8")[..., ::-1],
            )
            return self._loss_fn(image, gt_image)

        loss_and_grad_fn = mx.value_and_grad(loss_fn)

        for step, idx in enumerate(indices):
            self.step_count += 1
            logger.info(
                f"Epoch {epoch}, Step {step + 1}/{len(self.dataset)}: Rendering and computing loss for image {idx}."
            )
            gt_image, camera = self.dataset[idx]

            cv2.imwrite(
                "gt_image.jpg",
                (np.array(gt_image) * 255).astype("uint8")[..., ::-1],
            )

            loss, grads = loss_and_grad_fn(self.params, camera, gt_image)

            mx.eval(loss)

            logger.info(f"Epoch {epoch}, Step {step + 1}/{len(self.dataset)}: Loss = {loss.item():.6f}")
            epoch_loss += loss.item()
            num_rendered += 1

            self.optimizer.update(self.params, grads)
            mx.eval(self.params, self.optimizer.state)

            # Accumulate spatial position gradient norms
            pos_grads = grads["positions"]
            grad_norms = mx.linalg.norm(pos_grads, axis=-1)
            self.pos_grad_accum = self.pos_grad_accum + grad_norms
            self.denom = self.denom + 1.0

            # Perform densification/pruning check every N steps
            if self.step_count % self.configuration.densification_interval == 0:
                self._refine_gaussians()

            logger.info(f"Gradients: {[g.mean().item() for g in grads.values()]}")

        return epoch_loss / num_rendered if num_rendered > 0 else 0.0

    def _refine_gaussians(self) -> None:
        N = self.params["positions"].shape[0]
        grads_norm = self.pos_grad_accum / mx.maximum(self.denom, 1.0)

        # 1. Identify Gaussians to prune
        should_prune = self.params["opacities"].squeeze() < self.configuration.prune_opacity_threshold

        # 2. Identify Gaussians to clone or split based on average gradient
        should_densify = grads_norm > self.configuration.densify_grad_threshold

        # We assume scales are stored as log-scales (standard in 3DGS)
        split_cond = mx.max(mx.exp(self.params["scales"]), axis=-1) > self.configuration.split_scale_threshold

        should_split = should_densify & split_cond & ~should_prune
        should_clone = should_densify & ~split_cond & ~should_prune

        # Convert masks to NumPy to bypass MLX boolean indexing limitations
        should_prune_np = np.array(should_prune)
        should_split_np = np.array(should_split)
        should_clone_np = np.array(should_clone)

        indices_seq_np = np.arange(N)
        keep_idx_np = indices_seq_np[~should_prune_np & ~should_split_np]
        clone_idx_np = indices_seq_np[should_clone_np]
        split_idx_np = indices_seq_np[should_split_np]

        # Convert back to MLX arrays
        keep_idx = mx.array(keep_idx_np, dtype=mx.int32)
        clone_idx = mx.array(clone_idx_np, dtype=mx.int32)
        split_idx = mx.array(split_idx_np, dtype=mx.int32)

        num_keep = keep_idx.shape[0]
        num_clone = clone_idx.shape[0]
        num_split = split_idx.shape[0]

        # Combine indices to index existing tensors
        new_indices = mx.concat([keep_idx, clone_idx, split_idx, split_idx], axis=0)

        split_start_1 = num_keep + num_clone
        split_end_1 = split_start_1 + num_split
        split_start_2 = split_end_1
        split_end_2 = split_start_2 + num_split

        new_params = {}
        for key, val in self.params.items():
            new_params[key] = val[new_indices]

        # Split execution and scale adjustment
        if num_split > 0:
            scale_reduction = mx.log(mx.array(1.6))
            new_params["scales"] = mx.concatenate(
                [
                    new_params["scales"][:split_start_1],
                    new_params["scales"][split_start_1:split_end_1] - scale_reduction,
                    new_params["scales"][split_start_2:split_end_2] - scale_reduction,
                ],
                axis=0,
            )

            # Sample random noise for positional offset
            noise1 = mx.random.normal(shape=(num_split, 3))
            noise2 = mx.random.normal(shape=(num_split, 3))

            split_scales = mx.exp(self.params["scales"][split_idx])
            scaled_noise1 = noise1 * split_scales
            scaled_noise2 = noise2 * split_scales

            split_quats = new_params["quaternions"][split_start_1:split_end_1]
            split_quats = split_quats / mx.linalg.norm(split_quats, axis=-1, keepdims=True)

            def rotate_by_quaternion(v: mx.array, q: mx.array) -> mx.array:
                w = q[:, 0:1]
                xyz = q[:, 1:]

                def cross_product(a, b):
                    return mx.stack(
                        [
                            a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1],
                            a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2],
                            a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0],
                        ],
                        axis=-1,
                    )

                cross_xyz_v = cross_product(xyz, v)
                return v + 2.0 * cross_product(xyz, cross_xyz_v + w * v)

            offset1 = rotate_by_quaternion(scaled_noise1, split_quats)
            offset2 = rotate_by_quaternion(scaled_noise2, split_quats)

            new_params["positions"] = mx.concatenate(
                [
                    new_params["positions"][:split_start_1],
                    new_params["positions"][split_start_1:split_end_1] + offset1,
                    new_params["positions"][split_start_2:split_end_2] + offset2,
                ],
                axis=0,
            )

        # Apply newly built parameter set
        self.params = new_params
        new_num_gaussians = self.params["positions"].shape[0]

        # Reset tracking registers for the modified shape
        self.pos_grad_accum = mx.zeros((new_num_gaussians,), dtype=mx.float32)
        self.denom = mx.zeros((new_num_gaussians,), dtype=mx.float32)

        # Recursive updater to process the nested optimizer states dynamically
        def update_state(state, indices, original_size):
            if isinstance(state, mx.array):
                if state.ndim > 0 and state.shape[0] == original_size:
                    return state[indices]
                return state
            elif isinstance(state, dict):
                return {k: update_state(v, indices, original_size) for k, v in state.items()}
            elif isinstance(state, list):
                return [update_state(v, indices, original_size) for v in state]
            return state

        self.optimizer.state = update_state(self.optimizer.state, new_indices, N)

        mx.eval(self.params, self.optimizer.state)

        logger.info(
            f"Adaptive density control complete: {N} -> {new_num_gaussians} Gaussians "
            f"(Pruned: {should_prune_np.sum()}, Cloned: {num_clone}, Split: {num_split})"
        )

    def _loss_fn(
        self,
        image: mx.array,
        gt_image: mx.array,
    ) -> mx.array:
        l1 = mx.mean(mx.abs(image - gt_image))
        ssim_loss = 1.0 - ssim(image, gt_image)

        return 0.8 * l1 + 0.2 * ssim_loss

    def _build_gaussian_collection(
        self,
        params: dict[str, mx.array],
    ) -> GaussianCollection:
        return GaussianCollection.from_tensors(
            positions=params["positions"],
            quaternions=params["quaternions"],
            scales=params["scales"],
            sh_coeffs=params["sh_coeffs"],
            opacities=params["opacities"],
        )
