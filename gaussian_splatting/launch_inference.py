import click
import yaml

from gaussian_splatting.structures.launchers.inference_launcher import InferenceConfig, InferenceLauncher
from gaussian_splatting.utils.logger import Logger

logger = Logger("LAUNCH_INFERENCE")


@click.command()
@click.argument(
    "config_path",
    type=click.Path(exists=True),
)
def main(config_path: str):
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    config = InferenceConfig(**config_dict)
    launcher = InferenceLauncher(config)

    launcher.run()


if __name__ == "__main__":
    main()
