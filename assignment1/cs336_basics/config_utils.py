import argparse
from datetime import UTC, datetime
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs"
DEFAULT_DATA_CONFIG = CONFIG_ROOT / "data" / "default.yaml"
DEFAULT_MODEL_CONFIG = CONFIG_ROOT / "model" / "default.yaml"
DEFAULT_TRAINING_CONFIG = CONFIG_ROOT / "training" / "default.yaml"

OmegaConf.register_new_resolver(
    "timestamp",
    lambda: datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S"),
    use_cache=True,
    replace=True,
)


def load_config(description: str) -> tuple[DictConfig, argparse.Namespace]:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_CONFIG,
        help=f"Data config file (default: {DEFAULT_DATA_CONFIG}).",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_CONFIG,
        help=f"Model config file (default: {DEFAULT_MODEL_CONFIG}).",
    )
    parser.add_argument(
        "--training",
        type=Path,
        default=DEFAULT_TRAINING_CONFIG,
        help=f"Training config file (default: {DEFAULT_TRAINING_CONFIG}).",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="OmegaConf dotlist overrides, for example training.batch_size=8.",
    )
    args = parser.parse_args()

    config_paths = (args.data, args.model, args.training)
    missing_paths = [str(path) for path in config_paths if not path.is_file()]
    if missing_paths:
        parser.error(f"Missing config file(s): {', '.join(missing_paths)}")

    config = OmegaConf.merge(
        *(OmegaConf.load(path) for path in config_paths),
        OmegaConf.from_dotlist(args.overrides),
    )
    OmegaConf.resolve(config)
    return config, args
