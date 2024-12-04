# Evaluate the saved models on the test set

import argparse
import os
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

@hydra.main(config_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), "conf"),
            config_name="config",
            version_base="1.1")
def main(config) -> None:
    model_name = config["model"]["save_name"]
    model_path = config["file_path"]
    model = SklearnWrapper.load(model_path, config=config)
    pl.seed_everything(config["seed"])

    dm = MODTabularDataModule(config=config)
    dm.setup()

    metrics, prob_metrics, example_details = [], [], defaultdict(list)
    y, y_pred, y_prob = [], [], []

    predict_proba = hasattr(model, "predict_proba")
    for idx, (features, label) in tqdm(enumerate(dm.test_dataloader()),
                                       desc=f"Evaluating {model_path}",
                                       total=len(dm.test_dataloader())):
        try:
            features_serialized = pd.DataFrame(features)

            if pd.isna(label[model.target_label]) or pd.isna(label[model.target_label].item()):
                raise ValueError(f"Skipping sample {idx} due to missing label.")

            example_y = label[model.target_label].item()
            example_y_pred = model.predict(features_serialized).item()

            # Append examples last so that if there's an error, it's consistent in the output.

            if predict_proba:
                y_prob_e = model.predict_proba(features_serialized)
                # Get the probability of the gt class
                y_prob.append(y_prob_e[0, 1].item())

            y.append(example_y)
            y_pred.append(example_y_pred)

            for k, v in features.items():
                example_details[k].append(v.item())
        except Exception as e:
            print(f"Skipping sample {idx} due to error: {e}")
            continue

    # Load X_test and convert into a dataframe
    y_mapper = lambda y_hat: y_hat
    if isinstance(model, SklearnClassificationWrapper):
        y_mapper = lambda y_hat: 1 if y_hat > 1 else 0

    X_train, y_train = dm.dataloader_to_dataframe(dm.train_dataloader(), target_label=model.target_label,
                                         map_y=y_mapper)
    X_val, y_val = dm.dataloader_to_dataframe(dm.val_dataloader(), target_label=model.target_label,
                                         map_y=y_mapper)
    X_comb = pd.concat([X_train, X_val])
    y_comb = np.concatenate([y_train, y_val])
    X_test, y_test = dm.dataloader_to_dataframe(dm.test_dataloader(), target_label=model.target_label,
                                        map_y=y_mapper)

    # Perform SHAP analysis
    fn = model.predict
    if isinstance(model, SklearnClassificationWrapper):
        fn = model.predict_proba
    explainer = shap.Explainer(fn, X_comb)
    shap_values = explainer(X_test)

    # Save the evaluation results for plotting
    result = EvaluationResult(
        y=torch.tensor(y),
        y_pred=torch.tensor(y_pred),
        y_prob=torch.tensor(np.array(y_prob)),
        example_details=example_details,
        model_name=model.save_name,
        shap_values=shap_values,
        ablation_config=model.ablation_config,
        horizon=model.horizon,
        seed=model.seed,
    )
    model_name = model_name.replace(".pkl", "")
    params = [model_name, model.ablation_config]
    if model.horizon is not None:
        params.append(str(model.horizon))
    if model.seed is not None:
        params.append(str(model.seed))
    save_name = "_".join(params) + "-evaluation.pkl"
    save_path = os.path.normpath(os.path.join(config["models"]["results"], save_name))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(result, f)
    print(f"Saved evaluation results to {save_path}")


if __name__ == "__main__":
    main()
