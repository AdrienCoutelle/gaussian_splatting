from plyfile import PlyData

from gaussian_splatting.structures.gaussian import GaussianCollection
from gaussian_splatting.utils.logger import Logger
from gaussian_splatting.utils.ply_loader import load_ply_gaussians

logger = Logger("PLY_HANDLER")


class PLYHandler:
    def __init__(
        self,
        file_path: str,
    ):
        self.data = PlyData.read(file_path)

    def get_gaussians(self) -> GaussianCollection:
        return load_ply_gaussians(self.data)

    def log_info(self) -> None:
        logger.info(f"PLY file info:\n{self.data}")
