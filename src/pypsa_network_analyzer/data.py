from pathlib import Path
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="../../configs", config_name="default_config")
def preprocess(cfg: DictConfig) -> None:
    """Preprocess data using config parameters."""

    print(cfg.api_key)

if __name__ == "__main__":
    preprocess()
