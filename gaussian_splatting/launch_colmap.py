import os

from gaussian_splatting.utils.ply.ply_loader import PLYLoader

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import click
import yaml

from gaussian_splatting.utils.logger import Logger
from gaussian_splatting.utils.scene_preprocessor import ScenePreprocessor, ScenePreprocessorConfig

logger = Logger("LAUNCH_COLMAP")


@click.command()
@click.argument(
    "config_path",
    type=click.Path(exists=True),
)
def main(config_path: str) -> None:
    with open(config_path) as file:
        config_dict = yaml.safe_load(file)

    config = ScenePreprocessorConfig(**config_dict)
    runner = ScenePreprocessor(config)
    runner.run()

    ply_loader = PLYLoader(runner.output_folder / config.points_filename)

    ply_loader.log_info()
    gaussian_collection = ply_loader.get_gaussians()
    logger.info(f"Extracted {len(gaussian_collection)} Gaussians from COLMAP output.")
    logger.info(f"Sample Gaussian: {gaussian_collection.to_list()[0]}")
    # TODO: Here are only 3 sh coefficients, I would like to have more

    # pycolmap C++ destructors segfault on macOS during interpreter shutdown;
    # os._exit bypasses Python GC cleanup to avoid it.
    os._exit(0)


if __name__ == "__main__":
    main()
