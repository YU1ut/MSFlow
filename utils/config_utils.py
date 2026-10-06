from pathlib import Path
from typing import Any, Iterable

from omegaconf import DictConfig, OmegaConf, open_dict


def _to_serializable(value: Any) -> Any:
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    if hasattr(value, "__dict__"):
        return {k: _to_serializable(v) for k, v in vars(value).items()}
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    return value


def save_args_yaml(args: Any, config_path: str) -> None:
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    config = OmegaConf.create(_to_serializable(args))
    OmegaConf.save(config=config, f=str(path), resolve=True)


def load_yaml_config(config_path: str) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        return {}

    loaded = OmegaConf.load(str(path))
    config = OmegaConf.to_container(loaded, resolve=True)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a dict config in {config_path}.")
    return config


def apply_config_to_cfg(
    cfg: DictConfig, config: dict[str, Any], keys: Iterable[str]
) -> DictConfig:
    with open_dict(cfg):
        for key in keys:
            if key in config:
                cfg[key] = config[key]
    return cfg
