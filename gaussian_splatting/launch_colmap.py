import click
import yaml

from gaussian_splatting.utils.colmap import ColmapConfig, ColmapRunner
from gaussian_splatting.utils.logger import Logger

logger = Logger("LAUNCH_COLMAP")


@click.command()
@click.argument(
    "config_path",
    type=click.Path(exists=True),
)
def main(config_path: str) -> None:
    with open(config_path) as file:
        config_dict = yaml.safe_load(file)

    config = ColmapConfig.from_dict(config_dict)
    runner = ColmapRunner(config)

    logger.info("Starting COLMAP reconstruction...")
    results = runner.run()

    logger.info("COLMAP reconstruction complete!")
    logger.info(f"PLY: {results.ply_path}")
    logger.info(f"Poses: {results.poses_path}")
    logger.info(f"Intrinsics: {results.intrinsics_path}")


if __name__ == "__main__":
    main()
