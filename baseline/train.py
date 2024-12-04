# Trainer for the baseline models

# Pretraining and evaluation
import argparse
import os
import sys
from datetime import datetime

import lightning.pytorch as pl
import numpy as np
import torch
import yaml
from lightning import Callback
from lightning.pytorch.callbacks import TQDMProgressBar, EarlyStopping
from lightning.pytorch.profilers import PyTorchProfiler, AdvancedProfiler


from data import MODTabularDataModule, MODTabularDataset
from sklearn_models import *

import hydra
from omegaconf import DictConfig, OmegaConf

import rootutils
rootutils.setup_root(__file__, pythonpath=True)


@hydra.main(config_path=os.path.join(os.path.dirname(os.getcwd()), "conf"),
            config_name="config",
            version_base="1.1")
def main(config: DictConfig) -> None:

    # Set the global seed
    pl.seed_everything(config["seed"])

    model = eval(config["model"]["class_name"])(config=config)

    debug_mode = hasattr(sys, "gettrace")

    if isinstance(model, pl.LightningModule):
        raise NotImplementedError("Add as needed")
    else:
        dm = MODTabularDataModule(config=config)
        dm.setup()
        model.fit(dm)
        model.save()

if __name__ == "__main__":
    main()

