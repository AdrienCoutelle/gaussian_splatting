import os
from typing import Annotated

import cv2
import mlx.core as mx
import mlx.optimizers as opt
import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

from gaussian_splatting.structures.camera import Camera
from gaussian_splatting.structures.dataset import GaussianSplattingDataset
from gaussian_splatting.structures.gaussian import GaussianCollection
from gaussian_splatting.structures.renderer.rasterizer import Rasterizer
from gaussian_splatting.structures.renderer.renderer import Renderer
from gaussian_splatting.utils.logger import Logger
from gaussian_splatting.utils.ply.ply_saver import PLYSaver
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
    grad_clip_norm: float = 1.0  # max gradient L2 norm; 0 = disabled


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

        os.makedirs(output_folder, exist_ok=True)

        tb_log_dir = os.path.join(output_folder, "tensorboard")
        self.tensorboard_writer = TensorBoardWriter(tb_log_dir)
        logger.info(f"TensorBoard logs: {tb_log_dir}")
        mx.set_cache_limit(0)

        self.renderer.config.max_gaussians_per_batch = min(self.renderer.config.max_gaussians_per_batch, 128)
        self.renderer.config.max_gaussians_per_tile = min(self.renderer.config.max_gaussians_per_tile, 512)

        self.renderer.rasterizer = Rasterizer(
            gaussian_extent=self.renderer.config.gaussian_extent,
            tile_size=self.renderer.config.tile_size,
            max_gaussians_per_batch=self.renderer.config.max_gaussians_per_batch,
        )

        self.params = {
            "positions": mx.array(gaussians_collection.positions, dtype=mx.float32),
            "quaternions": mx.array(gaussians_collection.quaternions, dtype=mx.float32),
            "scales": mx.array(gaussians_collection.scales, dtype=mx.float32),
            "sh_coeffs": mx.array(gaussians_collection.sh_coeffs, dtype=mx.float32),
            "opacities": mx.array(gaussians_collection.opacities, dtype=mx.float32),
        }

        self.optimizer = opt.Adam(learning_rate=configuration.learning_rate)

        # Initialize the gradient function with gradient checkpointing
        def render_from_params(params: dict[str, mx.array], camera: Camera) -> mx.array:
            gaussians = self._build_gaussian_collection(params)
            return self.renderer.render_tensor(camera=camera, gaussians=gaussians)

        self.checkpointed_render = mx.checkpoint(render_from_params)

        def loss_fn(params: dict[str, mx.array], camera: Camera, gt_image: mx.array, valid_idx: mx.array) -> mx.array:
            sliced_params = {k: v[valid_idx] for k, v in params.items()}
            rendered = self.checkpointed_render(sliced_params, camera)
            return mx.mean(mx.abs(rendered - gt_image))

        self.loss_and_grad_fn = mx.value_and_grad(loss_fn)

    def _build_gaussian_collection(self, params: dict[str, mx.array]) -> GaussianCollection:
        return GaussianCollection.from_tensors(
            positions=params["positions"],
            quaternions=params["quaternions"],
            scales=params["scales"],
            sh_coeffs=params["sh_coeffs"],
            opacities=params["opacities"],
        )

    def _load_training_image(
        self,
        image_name: str,
        target_height: int,
        target_width: int,
    ) -> mx.array | None:
        for ext in [".png", ".jpg", ".jpeg"]:
            path = os.path.join(self.dataset.images_folder_path, image_name + ext)
            if not os.path.exists(path):
                continue
            img = cv2.imread(path)
            if img is None:
                return None
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img.shape[0] != target_height or img.shape[1] != target_width:
                img = cv2.resize(img, (target_width, target_height))
            return mx.array(img / 255.0, dtype=mx.float16)
        return None

    def _save_checkpoint(
        self,
        epoch: int,
    ) -> None:
        checkpoint_path = os.path.join(self.output_folder, f"checkpoint_epoch_{epoch:04d}.ply")
        gaussians_collection = self._build_gaussian_collection(self.params)
        PLYSaver(checkpoint_path).save_gaussians(gaussians_collection)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

    def _cull_gaussians(self, camera: Camera, active_focal_length: float) -> mx.array:
        positions = self.params["positions"]
        r_world_to_camera = mx.array(camera.pose[:3, :3].T)
        camera_center = mx.array(camera.pose[:3, 3:4])

        mx.eval(positions)  # Evaluate coordinates before operations

        cam_positions = (r_world_to_camera @ (positions.T - camera_center)).T
        depths = -cam_positions[:, 2]

        depth_mask = (depths > 0.1) & (depths < 20.0)

        principal_point_x, principal_point_y = camera.principal_point
        means_2d_x = active_focal_length * (cam_positions[:, 0] / mx.maximum(depths, 1e-4)) + principal_point_x
        means_2d_y = -active_focal_length * (cam_positions[:, 1] / mx.maximum(depths, 1e-4)) + principal_point_y

        margin = 80.0
        frustum_mask = (
            (means_2d_x >= -margin)
            & (means_2d_x <= camera.width + margin)
            & (means_2d_y >= -margin)
            & (means_2d_y <= camera.height + margin)
        )

        valid_mask = depth_mask & frustum_mask
        valid_mask_np = np.array(valid_mask)
        valid_idx_np = np.where(valid_mask_np)[0]

        valid_idx = mx.array(valid_idx_np, dtype=mx.int32)
        mx.eval(valid_idx)

        max_gs = self.configuration.max_gaussians_per_step
        if max_gs > 0 and valid_idx.shape[0] > max_gs:
            perm = np.random.permutation(valid_idx.shape[0])[:max_gs]
            valid_idx = valid_idx[mx.array(perm, dtype=mx.int32)]
            mx.eval(valid_idx)

        return valid_idx

    def _log_visuals(self, epoch: int, camera: Camera, valid_idx: mx.array, gt_image: mx.array) -> None:
        gt_np = np.array(gt_image, dtype=np.float32)
        sliced_params = {k: v[valid_idx] for k, v in self.params.items()}
        rendered_np = np.clip(
            np.array(
                self.renderer.render_tensor(
                    camera=camera,
                    gaussians=self._build_gaussian_collection(sliced_params),
                ),
                dtype=np.float32,
            ),
            0.0,
            1.0,
        )
        self.tensorboard_writer.log_image("Render/Generated", rendered_np, epoch)
        self.tensorboard_writer.log_image("Render/GroundTruth", gt_np, epoch)

    def _run_evaluation(self, epoch: int, camera_item: tuple, scale: float) -> None:
        image_name, camera = camera_item
        render_h = max(1, round(camera.h * scale))
        render_w = max(1, round(camera.w * scale))

        gt_image = self._load_training_image(
            image_name=image_name,
            target_height=render_h,
            target_width=render_w,
        )
        if gt_image is None:
            return

        eval_camera = Camera(
            pose=camera.pose,
            focal_length=camera.focal_length * scale,
            width=render_w,
            height=render_h,
        )

        rendered = np.clip(
            np.array(
                self.renderer.render_tensor(
                    camera=eval_camera,
                    gaussians=self._build_gaussian_collection(self.params),
                ),
                dtype=np.float32,
            ),
            0.0,
            1.0,
        )
        gt_np = np.array(gt_image, dtype=np.float32)

        mse_val = float(np.mean((rendered - gt_np) ** 2))
        psnr_val = -10.0 * np.log10(mse_val) if mse_val > 0 else float("inf")
        ssim_val = ssim(
            gt_np,
            rendered,
            data_range=1.0,
            channel_axis=-1,
        )

        logger.info(f"Epoch {epoch} — Evaluation | MSE: {mse_val:.6f} | PSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f}")
        self.tensorboard_writer.log_scalar("Metrics/MSE", mse_val, epoch)
        self.tensorboard_writer.log_scalar("Metrics/PSNR", psnr_val, epoch)
        self.tensorboard_writer.log_scalar("Metrics/SSIM", ssim_val, epoch)

    def _run_epoch(self, epoch: int, camera_items: list, scale: float) -> float:
        indices = mx.random.permutation(len(camera_items)).tolist()
        accum_steps = self.configuration.gradient_accumulation_steps
        accumulated_grads = {k: mx.zeros_like(v) for k, v in self.params.items()}

        epoch_loss = 0.0
        num_rendered = 0

        for step, idx in enumerate(indices):
            image_name, camera = camera_items[idx]

            render_h = max(1, round(camera.h * scale))
            render_w = max(1, round(camera.w * scale))

            # Cap maximum resolution limits
            max_pixels = 400 * 300
            current_pixels = render_h * render_w
            if current_pixels > max_pixels:
                adjust_scale = np.sqrt(max_pixels / current_pixels)
                render_h = max(1, round(render_h * adjust_scale))
                render_w = max(1, round(render_w * adjust_scale))
                active_focal_length = camera.focal_length * scale * adjust_scale
            else:
                active_focal_length = camera.focal_length * scale

            render_camera = Camera(
                pose=camera.pose,
                focal_length=active_focal_length,
                width=render_w,
                height=render_h,
            )

            gt_image = self._load_training_image(
                image_name=image_name,
                target_height=render_h,
                target_width=render_w,
            )
            if gt_image is None:
                continue

            valid_idx = self._cull_gaussians(render_camera, active_focal_length)
            logger.info(
                f"Epoch {epoch} — Camera {idx + 1}/{len(camera_items)} | "
                f"Keeping {valid_idx.shape[0]} / {self.params['positions'].shape[0]} Gaussians"
            )

            if valid_idx.shape[0] == 0:
                continue

            loss, grads = self.loss_and_grad_fn(self.params, render_camera, gt_image, valid_idx)

            for k in self.params:
                accumulated_grads[k] = accumulated_grads[k] + grads[k] / accum_steps

            mx.eval(loss, accumulated_grads)
            loss_val = loss.item()

            global_step = (epoch - 1) * len(indices) + step
            self.tensorboard_writer.log_scalar("Loss/train_step", loss_val, global_step)

            self._log_visuals(epoch, render_camera, valid_idx, gt_image)

            is_last_step = step == len(indices) - 1
            if (step + 1) % accum_steps == 0 or is_last_step:
                clip_norm = self.configuration.grad_clip_norm
                if clip_norm > 0:
                    total_sq = sum(float(mx.sum(mx.square(g)).item()) for g in accumulated_grads.values())
                    global_norm = float(np.sqrt(total_sq))
                    if global_norm > clip_norm:
                        scale_factor = clip_norm / (global_norm + 1e-6)
                        accumulated_grads = {k: v * scale_factor for k, v in accumulated_grads.items()}

                self.optimizer.update(self.params, accumulated_grads)
                mx.eval(self.params, self.optimizer.state)

                accumulated_grads = {k: mx.zeros_like(v) for k, v in self.params.items()}
                mx.eval(accumulated_grads)
                logger.info("Executed backpropagation and updated optimizer.")

            epoch_loss += loss_val
            num_rendered += 1

            del loss, grads, gt_image, valid_idx
            mx.clear_cache()

        return epoch_loss / num_rendered if num_rendered > 0 else 0.0

    def run(self) -> None:
        logger.info(f"Starting training for {self.configuration.epochs} epochs over {len(self.dataset)} cameras.")

        camera_items = list(self.dataset.items)
        scale = self.configuration.render_scale

        for epoch in tqdm(range(1, self.configuration.epochs + 1), desc="Training"):
            avg_loss = self._run_epoch(epoch, camera_items, scale)

            logger.info(f"Epoch {epoch}/{self.configuration.epochs} completed — Avg Loss: {avg_loss:.6f}")
            self.tensorboard_writer.log_scalar("Loss/train_epoch", avg_loss, epoch)

            do_log_epoch = epoch % self.configuration.log_every_n_epochs == 0
            if do_log_epoch and len(camera_items) > 0:
                self._run_evaluation(epoch, camera_items[0], scale)

            if epoch % self.configuration.save_every_n_epochs == 0:
                self._save_checkpoint(epoch)

        self._save_checkpoint(self.configuration.epochs)
        self.tensorboard_writer.close()
