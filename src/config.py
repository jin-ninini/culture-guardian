import os

import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")
USERS_DIR = os.path.join(DATA_DIR, "users")
DEFAULT_INTENT_CONFIG = os.path.join(PROJECT_ROOT, "configs", "klue_roberta.yaml")

# Config keys holding paths relative to the project root
PATH_KEYS = ("train_path", "valid_path", "test_path", "save_dir")


def load_config(path: str) -> dict:
    """Load a YAML config, merge its base_config (relative to the file), resolve paths."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if "base_config" in cfg:
        base_path = os.path.join(os.path.dirname(os.path.abspath(path)), cfg["base_config"])
        with open(base_path, "r", encoding="utf-8") as f:
            base_cfg = yaml.safe_load(f)
        base_cfg.update(cfg)
        cfg = base_cfg
    for key in PATH_KEYS:
        if key in cfg:
            cfg[key] = os.path.join(PROJECT_ROOT, cfg[key])
    return cfg


def experiment_dir(cfg: dict) -> str:
    """Directory holding the checkpoint and outputs of one experiment."""
    return os.path.join(cfg["save_dir"], cfg["experiment_name"])
