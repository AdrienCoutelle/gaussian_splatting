import os

import click
import yaml

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # TODO@Adrien: Use a .env

from gaussian_splatting.structures.launchers.training_launcher import TrainingConfig, TrainingLauncher


@click.command()
@click.argument(
    "config_path",
    type=click.Path(exists=True),
)
def main(config_path: str) -> None:
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    config = TrainingConfig(**config_dict)
    launcher = TrainingLauncher(config)

    launcher.run()


if __name__ == "__main__":
    main()
