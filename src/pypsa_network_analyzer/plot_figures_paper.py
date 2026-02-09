import hydra
from omegaconf import DictConfig
from pathlib import Path


@hydra.main(version_base=None, config_name="default_config", config_path="../../configs")
def main(cfg: DictConfig):
    root = Path(cfg.paths.root)
    results_dir = root / "results_concat"
    figures_dir = root / "figures_paper"
    figures_dir.mkdir(exist_ok=True)


if __name__ == "__main__":
    main()
