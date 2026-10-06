import hashlib
import math
from typing import Any

import torch
import torch.distributed as dist
import wandb
from omegaconf import OmegaConf
from torchvision.utils import make_grid


def is_main_process():
    return dist.get_rank() == 0


def namespace_to_dict(config: Any) -> Any:
    if OmegaConf.is_config(config):
        return OmegaConf.to_container(config, resolve=True)
    if hasattr(config, "__dict__"):
        return {k: namespace_to_dict(v) for k, v in vars(config).items()}
    if isinstance(config, dict):
        return {k: namespace_to_dict(v) for k, v in config.items()}
    if isinstance(config, (list, tuple)):
        return [namespace_to_dict(v) for v in config]
    return config


def generate_run_id(exp_name):
    # https://stackoverflow.com/questions/16008670/how-to-hash-a-string-into-8-digits
    return str(int(hashlib.sha256(exp_name.encode("utf-8")).hexdigest(), 16) % 10**8)


def initialize(args, entity, exp_name, project_name):
    config_dict = namespace_to_dict(args)
    # wandb.login(key=os.environ["WANDB_KEY"])
    wandb_mode = "disabled" if entity == "your_wandb_entity" else "online"
    wandb.init(
        entity=entity,
        project=project_name,
        name=exp_name,
        config=config_dict,
        id=generate_run_id(exp_name),
        resume="allow",
        mode=wandb_mode,
    )


def log(stats, step=None):
    if is_main_process():
        wandb.log({k: v for k, v in stats.items()}, step=step)


def log_image(sample, step=None):
    if is_main_process():
        sample = array2grid(sample)
        wandb.log({f"samples": wandb.Image(sample), "train_step": step})


def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x, nrow=nrow, normalize=True, value_range=(-1, 1))
    x = (
        x.mul(255)
        .add_(0.5)
        .clamp_(0, 255)
        .permute(1, 2, 0)
        .to("cpu", torch.uint8)
        .numpy()
    )
    return x
