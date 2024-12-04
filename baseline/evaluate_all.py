# Evaluate the saved models on the test set

import argparse
import os
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List

import lightning.pytorch as pl
import numpy as np
import shap
import torch
import yaml
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from lightning import Callback
from lightning.pytorch.callbacks import TQDMProgressBar, EarlyStopping
from lightning.pytorch.profilers import PyTorchProfiler, AdvancedProfiler
import torchmetrics
import pickle

from data import MODTabularDataModule, MODTabularDataset
from sklearn_models import *

import hydra
from omegaconf import DictConfig, OmegaConf

import rootutils

rootutils.setup_root(__file__, pythonpath=True)

@dataclass
class EvaluationResult:
    model_name: str
    y: torch.Tensor
    y_pred: torch.Tensor
    y_prob: torch.Tensor
    example_details: Dict[str, List[float]]
    shap_values: shap.Explanation
    ablation_config: str
    horizon: int
    seed: int

@hydra.main(config_path=os.path.join(os.path.dirname(os.getcwd()), "conf"),
            config_name="config",
            version_base="1.1")
def main(config_base) -> None:
    subprocesses = []
    for model_name in os.listdir(config_base["models"]["saved"]):
        # Load the model
        if "._" in model_name:  # Skip hidden files
            continue
        # Fyi: some models could be too large to load into memory in the future.
        print(f"Evaluating {model_name}")

        model_path = os.path.normpath(os.path.join(config_base["models"]["saved"], model_name))
        model = SklearnWrapper.load(model_path, config=config_base)
        save_name = model.save_name

        # Is the latest version of the model newer than the latest evaluation?
        params = [model.save_name, model.ablation_config]
        if model.horizon is not None:
            params.append(str(model.horizon))
        if model.seed is not None:
            params.append(str(model.seed))
        save_name_fn = "_".join(params) + "-evaluation.pkl"
        save_path = os.path.normpath(os.path.join(config_base["models"]["results"], save_name_fn))
        if os.path.exists(save_path) and os.path.getmtime(save_path) >= os.path.getmtime(model_path):
            print(f"Model {save_name} already evaluated, skipping.")
            continue

        # Exec the evaluation
        fpath_escaped = "\"" + model_path.replace(" ", "\ ") + "\""
        eval_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "evaluate.py"))
        exec_str = f"python {eval_path} -m 'file_path={fpath_escaped}' model={save_name} ablation={model.ablation_config} horizon={model.horizon} seed={model.seed}"
        print(exec_str)
        if len(subprocesses) > 8:
            subprocesses[0].wait()
            subprocesses = subprocesses[1:]
        sp = subprocess.Popen(exec_str, shell=True)
        subprocesses.append(sp)

    # Wait for all subprocesses to finish
    for sp in subprocesses:
        sp.wait()


if __name__ == "__main__":
    main()
