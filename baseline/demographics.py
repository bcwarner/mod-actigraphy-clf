# Simple demogrpahics from the dataset

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
import matplotlib.pyplot as plt
import seaborn as sns

import rootutils
rootutils.setup_root(__file__, pythonpath=True)
from tabulate import tabulate

@hydra.main(config_path=os.path.join(os.path.dirname(os.getcwd()), "conf"),
            config_name="config",
            version_base="1.1")
def main(config: DictConfig) -> None:

    # Set the global seed
    pl.seed_everything(config["seed"])

    # Load the final data
    dm = MODTabularDataModule(config=config)
    dm.setup()

    # Load the actigraphy data as originally set.
    data_path = config["data"]["tabular_processed_path"]
    raw_data = pd.read_csv(data_path, index_col=0, na_values="NA")
    # Load the REDCap data as originally set.
    redcap_path = config["data"]["redcap"]
    redcap_data = pd.read_csv(redcap_path)

    # Patient count
    patient_count = redcap_data[config["data"]["redcap_features"]["user_id"]].nunique()
    final_patient_count = len(dm.dataset.individual_ids)

    print(tabulate([["Patient Count", patient_count, final_patient_count]], headers=["Enrolled", "Eligible"]))

    # Keep only the final patient data
    redcap_data = redcap_data[redcap_data[config["data"]["redcap_features"]["user_id"]].isin(dm.dataset.individual_ids)]

    # Other demographics
    for demographic in config["data"]["redcap_features"]["demographics"]:
        # Categorical data
        if redcap_data[demographic].nunique() < 20:
            categories = pd.DataFrame(redcap_data[demographic].value_counts())
            # Add a percentage column
            categories["Percent"] = categories[demographic] / categories[demographic].sum()
            categories["Percent"] = categories["Percent"].map("{:.2%}".format)
            print(categories.to_latex())
        else:
            print(tabulate([[demographic, redcap_data[demographic].mean(), redcap_data[demographic].std()]], headers=["Feature", "Mean", "Std. Dev."]))

    # Number of PTB births
    categories = pd.DataFrame(raw_data.drop_duplicates("ID")["PTB"].value_counts())
    categories["Percent"] = categories["PTB"] / categories["PTB"].sum()
    categories["Percent"] = categories["Percent"].map("{:.2%}".format)
    print(categories.to_latex())
    # Number of samples overall, and by trimester.
    samples = raw_data["ID"].value_counts().values
    samples_mean = samples.mean()
    samples_std = samples.std()
    print(f"Samples: ${samples_mean:.1f} \pm {samples_std:.1f}$")
    trimesters = [(0, 13 * 7 + 6), (14 * 7, 27 * 7 + 6), (28 * 7, 100 * 7)] # Arbitrary upper limit
    for i, (start, end) in enumerate(trimesters):
        tdata = raw_data[(raw_data["GA_Days"] >= start) & (raw_data["GA_Days"] <= end)]["ID"].value_counts().values

        samples_mean = tdata.mean()
        samples_std = tdata.std()
        print(f"Trimester {i} samples: ${samples_mean:.1f} \pm {samples_std:.1f}$")

    # Plot the distribution of the GAs
    # Use LaTeX
    plt.rc("text", usetex=True)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    raw_data["PTB"] = raw_data["PTB"].apply(lambda x: "PTB+" if x > 0 else "PTB-")
    sns.histplot(raw_data, x="GA_Days", bins=280, hue="PTB", ax=ax, edgecolor=None, discrete=True) #, kde=True, line_kws={"linewidth": 1, "alpha": 0.5})
    plt.xlabel("Gestational Age (Days)")
    plt.ylabel("Count")
    plt.title("Distribution of Actigraphy Samples")
    plt.tight_layout()
    plt.savefig(os.path.join(config.models.plots, "ga_distribution.pdf"))

if __name__ == "__main__":
    main()
