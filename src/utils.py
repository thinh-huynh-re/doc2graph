import os
from argparse import ArgumentParser
from typing import Any, Dict, List

import yaml

from src.paths import PATH


class AttrDict:
    def __init__(self, d: Dict[str, Any]):
        for k, v in d.items():
            if isinstance(k, (list, tuple)):
                setattr(self, k, [AttrDict(x) if isinstance(x, dict) else x for x in v])
            else:
                setattr(self, k, AttrDict(v) if isinstance(v, dict) else v)


class TrainConfig(AttrDict):
    def __init__(self, d: Dict[str, Any]):
        self.batch_size: int = d.get("batch_size", 15)
        self.epochs: int = d.get("epochs", 100000)
        self.lr: float = d.get("lr", 1e-3)
        self.weight_decay: float = d.get("weight_decay", 1e-4)
        self.val_size: float = d.get("val_size", 0.1)
        self.optimizer: str = d.get("optimizer", "Adam")
        self.scheduler: List[str] = d.get("scheduler", ["ReduceLROnPlateau"])
        self.stopper_metric: str = d.get("stopper_metric", "acc")
        self.seed: int = d.get("seed", 42)


def create_folder(folder_path: str) -> None:
    """create a folder if not exists

    Args:
        folder_path (str): path
    """
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    return


def get_train_config(filename: str) -> TrainConfig:
    with open(PATH.CONFIGS / f"{filename}.yaml") as fileobj:
        config = TrainConfig(yaml.safe_load(fileobj))
    return config


def get_config(filename: str) -> AttrDict:
    """get yaml config file

    Args:
        name (str): yaml file name without extension

    Returns:
        AttrDict: config
    """
    with open(PATH.CONFIGS / f"{filename}.yaml") as fileobj:
        config = AttrDict(yaml.safe_load(fileobj))
    return config


def project_tree() -> None:
    """Create the project tree folder"""
    create_folder(PATH.DATA)
    create_folder(PATH.OUTPUTS)
    create_folder(PATH.RUNS)
    create_folder(PATH.RESULTS)
    create_folder(PATH.TRAIN_SAMPLES)
    create_folder(PATH.TEST_SAMPLES)
    create_folder(PATH.CHECKPOINTS)
    return


def set_preprocessing(args: ArgumentParser) -> None:
    """Set preprocessings args

    Args:
        args (ArgumentParser):
    """
    with open(PATH.CONFIGS / "base.yaml") as fileobj:
        cfg_preprocessing = dict(yaml.safe_load(fileobj))
    cfg_preprocessing["FEATURES"]["add_geom"] = args.add_geom
    cfg_preprocessing["FEATURES"]["add_embs"] = args.add_embs
    cfg_preprocessing["FEATURES"]["add_hist"] = args.add_hist
    cfg_preprocessing["FEATURES"]["add_visual"] = args.add_visual
    cfg_preprocessing["FEATURES"]["add_eweights"] = args.add_eweights
    cfg_preprocessing["FEATURES"]["num_polar_bins"] = args.num_polar_bins
    cfg_preprocessing["LOADER"]["src_data"] = args.src_data
    cfg_preprocessing["GRAPHS"]["data_type"] = args.data_type
    cfg_preprocessing["GRAPHS"]["edge_type"] = args.edge_type
    cfg_preprocessing["GRAPHS"]["node_granularity"] = args.node_granularity

    with open(PATH.CONFIGS / "preprocessing.yaml", "w") as f:
        yaml.dump(cfg_preprocessing, f)
    return
