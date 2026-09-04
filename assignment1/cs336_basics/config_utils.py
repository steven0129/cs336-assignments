from collections.abc import Sequence
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs"


def compose_config(overrides: Sequence[str] | None = None) -> DictConfig:
    """Compose the application config outside a Hydra-decorated entry point."""
    with initialize_config_dir(config_dir=str(CONFIG_ROOT), version_base=None):
        return compose(config_name="config", overrides=list(overrides or ()))
