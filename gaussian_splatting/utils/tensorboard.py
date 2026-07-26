import numpy as np
from torch.utils.tensorboard import SummaryWriter


class TensorBoardWriter:
    """Thin wrapper around SummaryWriter with helpers for training scalars and images."""

    def __init__(self, log_dir: str) -> None:
        self.writer = SummaryWriter(log_dir=log_dir)

    def log_scalar(
        self,
        tag: str,
        value: float,
        step: int,
    ) -> None:
        self.writer.add_scalar(tag, value, global_step=step)

    def log_image(
        self,
        tag: str,
        image: np.ndarray,
        step: int,
    ) -> None:
        """Log an HWC uint8 or float32 [0,1] image."""
        if image.dtype != np.uint8:
            image = np.clip(image, 0.0, 1.0)
        # TensorBoard expects (C, H, W)
        self.writer.add_image(tag, image, global_step=step, dataformats="HWC")

    def close(self) -> None:
        self.writer.close()
