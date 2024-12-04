# Data loading/module for the feature engineering approach.

import os
import pickle
import sys
from collections import OrderedDict, defaultdict

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset
from lightning import pytorch as pl
from sklearn.preprocessing import StandardScaler
from typing import List
import pandas as pd
from omegaconf import DictConfig, OmegaConf
import pyActigraphy

from pyActigraphy.io.mtn import RawMTN
from tqdm import tqdm


class MODTimeseries(Dataset):
    """
    Assumptions:
    Features have already been engineered from raw data into a .csv file.
    """
    def __init__(self,
                 config: dict,
                 reset_cache: bool = False):
        super().__init__()
        self.config = config
        self.data_path = config["data"]["raw"]
        self.redcap_path = config["data"]["redcap"]

        # Load the REDCap data into a dataframe
        self.redcap_data = pd.read_csv(self.redcap_path)

        self.patient_to_folder = {}
        self.individual_ids = []
        self.examples = OrderedDict()
        self.reset_cache = reset_cache

        # Load the data
        folders = os.listdir(self.data_path)
        for folder in folders:
            if folder == ".DS_Store" or "." in folder: # Faster than testing if folder
                continue
            details = folder.split(" ")
            patient_id = details[0]

            self.individual_ids.append(patient_id)
            self.patient_to_folder[patient_id] = folder

    def load_example(self, patient_id: str):
        """
        Load a single example from the dataset.
        """
        # Does the example have a cache file?
        cache_fname = self.config["data"]["raw_config"]["cache_fname"]
        if patient_id in self.examples:
            return self.examples[patient_id]
        if os.path.exists(os.path.join(self.data_path, self.patient_to_folder[patient_id], cache_fname)) and not self.reset_cache:
            with open(os.path.join(self.data_path, self.patient_to_folder[patient_id], cache_fname), "rb") as f:
                user_example = pickle.load(f)
                self.examples[user_example["patient_id"]] = user_example
                return user_example
        else:
            folder = self.patient_to_folder[patient_id]
            details = folder.split(" ")
            patient_id = details[0]
            birth_status = details[1]
            written_term_status = details[2] if len(details) > 2 else None
            birth_date = details[3] if len(details) > 3 else None
            self.patient_to_folder[patient_id] = folder

            # Find the estimated delivery date, select by ID, then pick first value in dataframe
            redcap_fns = self.config["data"]["redcap_features"]
            estimated_date = self.redcap_data[self.redcap_data[redcap_fns["user_id"]] == int(patient_id)][
                redcap_fns["estimated_delivery_date"]].values[0]
            actual_date = self.redcap_data[self.redcap_data[redcap_fns["user_id"]] == int(patient_id)][
                redcap_fns["actual_delivery_date"]].values[0]

            # Difference between estimated and actual delivery date
            actual_diff = (pd.to_datetime(actual_date) - pd.to_datetime(estimated_date)).days


            # Load each of the actigraphy data files
            extension = self.config["data"]["raw_config"]["extension"]
            actigraphy_files = [os.path.join(self.data_path, folder, fn) for fn in
                                os.listdir(os.path.join(self.data_path, folder)) if fn.endswith(extension)]

            # Load each file from pyActigraphy
            # Then separate into different nights centered around midnight of each

            sub_examples = {}
            for file in actigraphy_files:
                act_file: RawMTN = pyActigraphy.io.read_raw_mtn(os.path.join(self.data_path, folder, file))
                time = act_file.data.index
                light = torch.tensor(act_file.data.values)
                motion = torch.tensor(act_file.raw_light.data.values).reshape(-1)

                # Find every instance of midnight (or close to it)
                for mid_i, t in enumerate(time):
                    if t.hour == 0 and t.minute == 0:
                        # Bottom range, noon previous day:
                        prev_time_bound = t - pd.Timedelta(hours=12)
                        # Top range, noon next day:
                        next_time_bound = t + pd.Timedelta(hours=12)
                        # Find the indices that fall within this range
                        indices = np.where((time >= prev_time_bound) & (time < next_time_bound))[0]
                        # Get the light and motion data for this range
                        light_range = light[indices]
                        motion_range = motion[indices]
                        # Calculate days before estimated/actual delivery
                        days_before_actual = (t - pd.to_datetime(actual_date)).days
                        days_before_estimated = (t - pd.to_datetime(estimated_date)).days
                        # Get the timedeltas centered around midnight
                        time_deltas = time[indices] - t
                        # Add to examples
                        string_name = f"{days_before_estimated}"
                        sub_examples[string_name] = {
                            "light": light_range,
                            "motion": motion_range,
                            "time": time_deltas,
                            "days_before_actual": days_before_actual,  # (label)
                            "days_before_estimated": days_before_estimated, # (feature)
                        }

            # Sort the sub_examples so that it's ascending numerically
            sub_examples = OrderedDict(sorted(sub_examples.items(), key=lambda x: int(x[0])))

            # Combine all examples into one
            user_example = {
                "patient_id": patient_id,
                "birth_status": birth_status,
                "actual_diff": actual_diff,
                "examples_len": len(sub_examples),
                "examples": sub_examples,
            }
            # Save in cache
            with open(os.path.join(self.data_path, folder, "cache.pkl"), "wb") as f:
                pickle.dump(user_example, f)

            self.examples[len(self.examples)] = user_example
            return user_example

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        if isinstance(index, str):
            return self.load_example(index)
        return self.load_example(self.individual_ids[index])

class MODTimeseriesModule(pl.LightningDataModule):
    """
    Data module for the feature engineering approach.
    Splits will be performed by patient IDs.
    """
    def __init__(self,
                 config: DictConfig = None,
                 reset_cache: bool = False):
        super().__init__()
        self.batch_size = config["data"]["batch_size"]
        self.num_workers = config["data"]["num_workers"] if not hasattr(sys, "gettrace") else 0
        self.config = config
        self.reset_cache = reset_cache

    def setup(self, stage=None):
        """
        Load data from disk.
        """

        # Load the data
        self.dataset = MODTimeseries(self.config,
                                     reset_cache=self.reset_cache)

        # Double check that seed is set
        pl.seed_everything(self.config["seed"])

        # Split the data by patient ID
        train_patient_ids, val_patient_ids, test_patient_ids = torch.utils.data.random_split(
            self.dataset.individual_ids,
            lengths=[
                self.config["data"]["split"]["train"],
                self.config["data"]["split"]["val"],
                self.config["data"]["split"]["test"],
            ],
        )

        self.train_indices = train_patient_ids
        self.val_indices = val_patient_ids
        self.test_indices = test_patient_ids

        self.train_dataset = torch.utils.data.Subset(self.dataset, self.train_indices)
        self.val_dataset = torch.utils.data.Subset(self.dataset, self.val_indices)
        self.test_dataset = torch.utils.data.Subset(self.dataset, self.test_indices)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
