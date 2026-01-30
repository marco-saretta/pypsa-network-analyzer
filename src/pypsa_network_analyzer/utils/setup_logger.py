import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import sys


def setup_logger(
    log_dir: str | Path = "./logs",
    name: str = "pypsa_network_analyzer",
) -> logging.Logger:
    """Setup logger with file and console output."""
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if logger.handlers:
        return logger
    
    # File handler
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    file_handler = RotatingFileHandler(
        log_dir / "run.log",
        maxBytes=10_000_000,
        backupCount=3,
    )
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    )
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(
        logging.Formatter("%(levelname)s | %(message)s")
    )
    logger.addHandler(console_handler)
    
    return logger