import hydra
from omegaconf import DictConfig

from eval.eval_msflow_263 import run


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="eval_msflow_xyz",
)
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
