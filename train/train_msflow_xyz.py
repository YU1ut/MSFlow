import hydra
from omegaconf import DictConfig

from train.train_msflow_263 import run


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="train_msflow_xyz",
)
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
