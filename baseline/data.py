# Data loading/module for the feature engineering approach.

import os
import pickle
import sys
from collections import OrderedDict, defaultdict

import numpy as np
import torch
import yaml
from pandas.core.dtypes.common import is_numeric_dtype
import re
from regex import regex
from sts_select.mrmr import MRMRBase
from torch.utils.data import Dataset
from lightning import pytorch as pl
from sklearn.preprocessing import StandardScaler
from typing import List
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sts_select.target_sel import TopNSelector
from sts_select.scoring import SentenceTransformerScorer

class MODTabularDataset(Dataset):
    """
    Assumptions:
    Features have already been engineered from raw data into a .csv file.
    """
    def __init__(self,
                 config: dict,):
        super().__init__()
        self.config = config
        self.data_path = config["data"]["tabular_processed_path"]
        self.raw_data = pd.read_csv(self.data_path, index_col=0,
                                    na_values="NA")

        if self.config["ablation"]["tabular_features"]:
            # Reset the index
            self.raw_data.reset_index(drop=True, inplace=True)

            # Find all columns that might be datetimes, and convert them to timestamps.
            raw_copy = self.raw_data.copy()
            for column in config["data"]["tabular_features"]["dt_features"]:
                res = raw_copy[column]
                try:
                    res = pd.to_datetime(self.raw_data[column]).astype(np.int64) // 10 ** 9
                    # Write a warning with the specific column name.
                except UserWarning as e:
                    raise UserWarning(f"Column {column} was converted to a timestamp.")
                except Exception as e:
                    pass
                raw_copy[column] = res

            # Drop all features listed in the config
            raw_copy.drop(columns=config["data"]["tabular_features"]["drop_features"], inplace=True)

            self.raw_data = raw_copy

            self.id_names: List[str] = config["data"]["tabular_features"]["id_names"]
            self.main_id: str = config["data"]["tabular_features"]["main_id"]
            self.label_names: List[str] = config["data"]["tabular_features"]["regression_label"] + \
                                          config["data"]["tabular_features"]["classification_label"]

            horizon_name = config["data"]["tabular_features"]["horizon_id"]
            self.feature_names = list(set(self.raw_data.columns.tolist()).difference(
                set(self.id_names).union(set(self.label_names).union(set([horizon_name])))
            ))

            # Copy the main patient ID to a new DF
            new_df = self.raw_data[[self.main_id] + self.label_names].copy()
            # Keep the first instance of each patient ID
            new_df.drop_duplicates(subset=[self.main_id], inplace=True)

            # Drop features before the horizon
            self.raw_data = self.raw_data[self.raw_data[horizon_name] <= self.config["horizon"]]
            # Drop the horizon features
            self.raw_data = self.raw_data.drop(columns=[horizon_name])

            for column in self.feature_names:
                # Change to an absolute value if specified
                if "dd" in column and ("dd_abs" in config["ablation"] and config["ablation"]["dd_abs"]):
                    gb_apply = lambda x: np.abs(x)
                else:
                    gb_apply = lambda x: x

                group_by = self.raw_data.groupby(config["data"]["tabular_features"]["main_id"])[column]
                mean = group_by.apply(lambda x: np.mean(gb_apply(x)))
                std = group_by.apply(lambda x: np.std(gb_apply(x)))

                # Rename the columns
                mean.rename(f"{column}_mean", inplace=True)
                std.rename(f"{column}_std", inplace=True)
                # Add to the new DF
                new_df = new_df.join(mean, on=self.main_id)
                new_df = new_df.join(std, on=self.main_id)

            temp_feature_names = self.feature_names.copy()
            for column in temp_feature_names:
                self.feature_names.extend([f"{column}_mean", f"{column}_std"])
                self.feature_names.remove(column)

            # Drop the rows where the regression label is < 0
            new_df = new_df[new_df[config["data"]["tabular_features"]["regression_label"][0]] >= 0]
            self.raw_data = new_df
        else:
            # Dummy table with only the main ID and the labels.
            self.id_names: List[str] = config["data"]["tabular_features"]["id_names"]
            self.main_id: str = config["data"]["tabular_features"]["main_id"]
            self.label_names: List[str] = config["data"]["tabular_features"]["regression_label"] + \
                                            config["data"]["tabular_features"]["classification_label"]
            self.raw_data = self.raw_data[self.id_names + self.label_names]
            # Aggregate the data by the main ID.
            self.raw_data = self.raw_data.groupby(self.main_id).last().reset_index()
            # Keep rows where the regression label is >= 0
            self.raw_data = self.raw_data[self.raw_data[config["data"]["tabular_features"]["regression_label"][0]] >= 0]
            self.feature_names = []

        self.redcap_path = config["data"]["redcap"]
        if config["ablation"]["sts_features"]:
            # Load the REDCap data.
            self.redcap_data = pd.read_csv(self.redcap_path)
            # Aggregate it down to main ID by last appearing value.
            self.redcap_data = self.redcap_data.groupby(config["data"]["redcap_features"]["user_id"]).last().reset_index()
            redcap_record_id = self.redcap_data[config["data"]["redcap_features"]["user_id"]].copy()

            # Load the REDCap feature names.
            self.redcap_features_path = config["data"]["redcap_feature_names"]["file"]
            self.redcap_features = pd.read_csv(self.redcap_features_path)

            # Find the last index of last_form, then drop everything after that.

            # Map the REDCap feature names to natural language names.
            # Set the Section Header to the next preceding value if it's NaN
            self.redcap_features["Section Header"] = self.redcap_features["Section Header"].fillna(method="ffill")

            # Then combine Section Header with Field Label, unless Section Header is NaN.
            self.redcap_features["Field Label"] = self.redcap_features["Section Header"].fillna("") + " - " + \
                                                    self.redcap_features["Field Label"].fillna("")

            redcap_feature_map = self.redcap_features.set_index("Variable / Field Name")["Field Label"].to_dict()
            regex_rep = regex.compile(r"(<[^<]+?>|\n|\r)")
            redcap_feature_map = {k: regex_rep.sub("", v) for k, v in redcap_feature_map.items()}

            last_form_index = self.redcap_features[self.redcap_features["Form Name"] == config["data"]["redcap_feature_names"]["last_form"]].index[-1]
            # Drop everything after the last form.
            to_drop = self.redcap_features.iloc[last_form_index + 1:]["Variable / Field Name"].tolist()

            # If the field is categorical, then the "Variable / Field Name" will include __1, __2, etc.
            # Split the feature names into individual categories since they are one-hot in the actual data.
            redcap_feature_map_one_hot = {}
            for k, v in redcap_feature_map.items():
                cat_value = self.redcap_features[self.redcap_features["Variable / Field Name"] == k]["Choices, Calculations, OR Slider Labels"].item()
                # Add the choices to the feature map
                droppable = self.redcap_features[self.redcap_features["Variable / Field Name"] == k]["Section Header"].item() in config["data"]["redcap_feature_names"]["exclude_sections"]
                droppable = droppable or k in to_drop # For one-hot encoded features
                if not pd.isna(cat_value) and "|" in cat_value:
                    # Split the choices
                    choices = cat_value.split("|")
                    # Split choices by index, value
                    choices = {int(x[0].strip()): x[1].strip() for x in [y.split(",") for y in choices]}
                    # Remove any HTML fragments and newlines from the Section Header or Field Label
                    choices = {k1: regex_rep.sub("", v1) for k1, v1 in choices.items()}
                    for i, choice in choices.items():
                        redcap_key = f"{k}___{i}"
                        if droppable:
                            to_drop.append(redcap_key)
                        else:
                            redcap_feature_map_one_hot[redcap_key] = f"{v} ({choice})"

                # Default case: no choices, just add the feature.
                # Always add, not always one-hot encoded.
                if droppable:
                    to_drop.append(k)
                else:
                    redcap_feature_map_one_hot[k] = v


            redcap_feature_map = redcap_feature_map_one_hot
            to_drop = list(set(to_drop) - set(config["data"]["tabular_features"]["redcap_include"]))
            self.redcap_data.drop(columns=to_drop, inplace=True, errors="ignore")

            # Now sum the data grouped by the sum_prefix.
            sum_prefix = regex.compile(config["data"]["tabular_features"]["sum_prefix"], re.IGNORECASE)
            sum_agg = defaultdict(pd.Series)
            to_drop = []
            for column in self.redcap_data.columns:
                match: re.Match = sum_prefix.match(column)
                if match and is_numeric_dtype(self.redcap_data[column]):
                    # Replace None with np.nan
                    na_temp = self.redcap_data[column].fillna(np.nan)
                    sum_agg[f"deliv1{match.group(2)}"] = sum_agg[f"deliv1{match.group(2)}"].add(na_temp, fill_value=0)
                    to_drop.append(column)
            self.redcap_data.drop(columns=to_drop, inplace=True)
            self.redcap_data = pd.concat([self.redcap_data, pd.DataFrame(sum_agg)], axis=1)

            # Avoid renaming for now
            #self.redcap_data.rename(columns=redcap_feature_map, inplace=True)

            self.redcap_data = self.redcap_data.fillna(np.nan)

            # Map the feature names to their index in the data.
            presel_indexes = [self.redcap_data.columns.get_loc(k) for k in (config["data"]["tabular_features"]["redcap_include"] + list(sum_agg.keys()))]

            # Drop ambiguous columns
            presel_indexes = [i for i in presel_indexes if isinstance(i, int)]

            # Select the best features with mRMR using STS.
            self.scorer = MRMRBase(
                    SentenceTransformerScorer(self.redcap_data.values,
                        np.zeros_like(self.redcap_data.values[0, :]), # Dummy y
                        X_names=self.redcap_data.columns.tolist(),
                        y_names=config["data"]["redcap_feature_names"]["y_names"],
                        model_path=config["data"]["sts_scorer"],
                        cache=config["data"]["redcap_feature_names"]["cache"],
                        verbose=1,
                    ),
                    n_features=config["data"]["redcap_feature_names"]["n_features"] + len(presel_indexes),
                    preselected_features=presel_indexes,
                )

            # Select the best features
            intermediate_values = self.scorer.fit(self.redcap_data, []).transform(self.redcap_data.values)
            sel_column_names = [self.redcap_data.columns[i] for i in self.scorer.sel_features]
            self.redcap_data = pd.DataFrame(intermediate_values, columns=sel_column_names)
            # Add the record ID back in.
            self.redcap_data[config["data"]["redcap_features"]["user_id"]] = redcap_record_id
            # Rename the ID column to match the main ID.
            self.redcap_data.rename(columns={config["data"]["redcap_features"]["user_id"]: self.main_id}, inplace=True)
            # Try and coerce all object types to floats, delete if not possible.
            col_to_drop = []
            for column in self.redcap_data.columns:
                try:
                    self.redcap_data[column] = self.redcap_data[column].astype(float)
                except ValueError as e:
                    col_to_drop.append(column)
            self.redcap_data.drop(columns=col_to_drop, inplace=True)
            self.raw_data = self.raw_data.join(self.redcap_data.set_index(self.main_id), on=self.main_id)
            self.feature_names.extend(set(sel_column_names) - set(col_to_drop))

        if "nulliparous_only" in config["ablation"] and config["ablation"]["nulliparous_only"]:
            # Load the REDCap data again, get the nulliparous only.
            nulliparous_ft_name = config["data"]["tabular_features"]["nulliparous_id"]
            self.redcap_data_copy = pd.read_csv(self.redcap_path)
            self.redcap_data_copy = self.redcap_data_copy.groupby(config["data"]["redcap_features"]["user_id"]).last().reset_index()
            self.redcap_data_copy = self.redcap_data_copy[self.redcap_data_copy[nulliparous_ft_name] == 1]
            self.raw_data = self.raw_data[self.raw_data[self.main_id].isin(self.redcap_data_copy[config["data"]["redcap_features"]["user_id"]])]

            # NA features that do not make sense for nulliparous patients.
            for column in config["data"]["tabular_features"]["nulliparous_exclude"]:
                self.raw_data[column] = np.nan


        # TODO: Clean this up
        #self.feature_names = list(set(self.raw_data.columns.tolist()) - set(self.id_names) - set(self.label_names))

        self.examples = OrderedDict()
        self.patient_id_to_indices = defaultdict(list) # Maps patient ID to indices in the dataset.
        self.indices_to_patient_id = OrderedDict() # Maps indices in the dataset to patient ID.
        # Go through and nicely collect into examples.
        for index, row in self.raw_data.iterrows():
            # Get the patient ID
            patient_id = row[self.main_id]
            # Get the features
            features = row[self.feature_names].to_dict()
            # Get the labels
            label = row[self.label_names].to_dict()
            # Add to the examples dictionary
            self.examples[index] = (features, label)
            self.patient_id_to_indices[patient_id].append(index)
            self.indices_to_patient_id[index] = patient_id
        self.individual_ids = list(self.patient_id_to_indices.keys())

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]

class MODTabularDataModule(pl.LightningDataModule):
    """
    Data module for the feature engineering approach.
    Splits will be performed by patient IDs.
    """
    def __init__(self,
                 config: DictConfig = None):
        super().__init__()
        self.batch_size = config["data"]["batch_size"]
        self.num_workers = config["data"]["num_workers"] if not hasattr(sys, "gettrace") else 0
        self.config = config

    def setup(self, stage=None):
        """
        Load data from disk.
        """

        # Load the data
        self.dataset = MODTabularDataset(self.config)

        # Double check that seed is set
        pl.seed_everything(self.config["seed"])
        test_len = int(self.config["data"]["split"]["test"] * len(self.dataset))
        val_len = int(self.config["data"]["split"]["val"] * len(self.dataset))
        train_len = int(len(self.dataset) - test_len - val_len)
        # Split the data by patient ID
        train_patient_ids, val_patient_ids, test_patient_ids = torch.utils.data.random_split(
            self.dataset.individual_ids,
            lengths=[
                train_len,
                val_len,
                test_len,
            ],
        )

        # Get the indices for each patient ID
        self.train_indices = []
        self.val_indices = []
        self.test_indices = []


        # Perform splitting
        for patient_id in train_patient_ids:
            self.train_indices.extend(self.dataset.patient_id_to_indices[patient_id])
        for patient_id in val_patient_ids:
            self.val_indices.extend(self.dataset.patient_id_to_indices[patient_id])
        for patient_id in test_patient_ids:
            self.test_indices.extend(self.dataset.patient_id_to_indices[patient_id])

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

    def dataloader_to_dataframe(self, dl, target_label, map_y: callable = None):
        def debatch(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()[0]
            else:
                return x[0]

        X = []
        y = []
        for features, label in dl:
            X.append({k: debatch(v) for k, v in features.items()})
            y_val = debatch(label[target_label])
            # TODO: Remove this once the data is cleaned.
            if pd.isna(y_val):
                y_val = 0  # Defaults to zero
            if map_y is not None:
                y_val = map_y(y_val)
            y.append(y_val)
        X = pd.DataFrame(X)
        X = X.reindex(sorted(X.columns), axis=1)
        y = np.array(y)
        return X, y