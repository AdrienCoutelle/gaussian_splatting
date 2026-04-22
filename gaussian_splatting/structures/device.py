import torch

from gaussian_splatting.utils.logger import Logger

logger = Logger("DEVICE")


class Device:
    device = None

    @classmethod
    def get(cls) -> torch.device:
        if cls.device is not None:
            return cls.device

        torch.set_default_dtype(torch.float32)

        if torch.backends.mps.is_available():
            cls.device = torch.device("mps")
            logger.info("Using device: MPS (Apple Silicon)")
        elif torch.cuda.is_available():
            cls.device = torch.device("cuda")
            logger.info("Using device: CUDA")
        else:
            cls.device = torch.device("cpu")
            logger.info("Using device: CPU")

        return cls.device
