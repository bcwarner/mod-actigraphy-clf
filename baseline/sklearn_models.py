import copy
import os
import pickle

import numpy as np
import pandas as pd
import sklearn
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer, OrdinalEncoder

import lightning as pl
import skops.io as skio
import xgboost as xgb

import rootutils
rootutils.setup_root(__file__, pythonpath=True)


from baseline.data import MODTabularDataModule


class SklearnWrapper(object):
    def __init__(self,
                 config: DictConfig,
                 ):
        self.config = config
        self.model = None
        self.pipeline = None
        self.target_label = None
        self.ablation_config = None
        self.save_name = None
        self.horizon = None

    def fit(self,
            dm: MODTabularDataModule):
        """
        Wrapper for fitting the model, takes in a Lightning DataModule.
        :return:
        """

        # Load all data from the trainer and collate into an X and y dataframe.
        y_mapper = lambda y: y
        if isinstance(self, SklearnClassificationWrapper):
            y_mapper = lambda y: 1 if y > 0 else 0

        X_train, y_train = dm.dataloader_to_dataframe(dm.train_dataloader(), self.target_label, y_mapper)
        X_val, y_val = dm.dataloader_to_dataframe(dm.val_dataloader(), self.target_label, y_mapper)

        X = pd.concat([X_train, X_val])
        y = np.concatenate([y_train, y_val])
        # No need to concatenate y, it's already a list.

        # Sort the columns alphabetically
        X: pd.DataFrame = X.reindex(sorted(X.columns), axis=1)

        if self.pipeline is None:
            # Construct the X DataFrameMapper
            numerical = []
            categorical = []
            for feature, dtype in zip(X.columns, X.dtypes):
                if dtype == np.dtype("O"):
                    categorical.append(feature)
                else:
                    numerical.append(feature)

            self.model.verbose = 2
            self.ct = ColumnTransformer(
                [("numerical", StandardScaler(), numerical),
                 # Ordinal in case we didn't see all the categories in the training set.
                 ("categorical", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categorical)]
            )

            pipeline_steps = [
                ("mapper", self.ct),
                ("imputer", sklearn.impute.SimpleImputer()),  # TODO: Reassess later.
            ]

            #if self.config["ablation"]["feature_selection"]:
            #    pipeline_steps.append(("feature_selector", sklearn.feature_selection.SelectFromModel(estimator=sklearn.linear_model.LogisticRegression(), threshold="0.25*mean")),)

            pipeline_steps.append(("model", self.model))

            self.pipeline = Pipeline(pipeline_steps)

        h_grid = OmegaConf.to_container(self.config["model"]["hyperparams"], resolve=True)
        # Convert it so that it's compatible with the pipeline
        h_grid = {f"model__{k}": [v] if not isinstance(v, list) else v for k, v in h_grid.items()}

        # Use CV to find the best hyperparameters
        gs = sklearn.model_selection.GridSearchCV(
            self.pipeline,
            param_grid=h_grid,
            cv=5,
            scoring="roc_auc" if isinstance(self, SklearnClassificationWrapper) else "neg_mean_squared_error",
            verbose=2,
        )
        gs.fit(X, y)
        self.pipeline = gs.best_estimator_


    def save(self):
        save_folder = os.path.normpath(self.config["models"]["saved"])
        save_name = (self.config["model"]["save_name"] + "_"
                     + self.config["ablation"]["name"] + "_"
                     + str(self.config["horizon"]) + "_"
                     + str(self.config["seed"]))
        save_path = os.path.join(save_folder, save_name + ".pkl")
        os.makedirs(save_folder, exist_ok=True)
        # Copy self without the config
        skio.dump({
            "pipeline": self.pipeline,
            "target_label": self.target_label,
            "class_name": self.__class__.__name__,
            "ablation_config": self.config["ablation"]["name"],
            "save_name": self.config["model"]["save_name"],
            "horizon": self.config["horizon"],
            "seed": self.config["seed"],
        }, open(save_path, "wb"))

    @classmethod
    def load(cls,
             full_path: str,
             config: DictConfig = None):
        data = skio.load(full_path, trusted=True)
        model = eval(data["class_name"])(config=config)
        model.class_name = data["class_name"]
        model.pipeline = data["pipeline"]
        model.target_label = data["target_label"]
        model.ablation_config = data["ablation_config"]
        model.save_name = data["save_name"]
        model.horizon = data["horizon"] if "horizon" in data else config["horizon"]
        model.seed = data["seed"] if "seed" in data else config["seed"]
        return model

    def __getattr__(self, item):
        return self.pipeline.__getattribute__(item)

    def __call__(self, *args, **kwargs):
        return self.pipeline.__call__(*args, **kwargs)


class SklearnRegressionWrapper(SklearnWrapper):
    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        if not self.target_label:
            self.target_label = self.config["data"]["tabular_features"][f"regression_label"][0]


class SklearnClassificationWrapper(SklearnWrapper):
    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        if not self.target_label:
            self.target_label = self.config["data"]["tabular_features"][f"classification_label"][0]


# ============== Regressors =================
class LinearSVMWrapper(SklearnRegressionWrapper):
    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.model = sklearn.svm.LinearSVR()

class SVMWrapper(SklearnRegressionWrapper):
    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.model = sklearn.svm.SVR()

class XGBoostRegressionWrapper(SklearnRegressionWrapper):
    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.model = xgb.XGBRegressor()


# =================== Classifiers ===================
class GaussianNBWrapper(SklearnClassificationWrapper):
    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.model = sklearn.naive_bayes.GaussianNB()


class XGBoostClassifierWrapper(SklearnClassificationWrapper):
    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.model = xgb.XGBClassifier()


class LogisticRegressionWrapper(SklearnClassificationWrapper):
    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.model = sklearn.linear_model.LogisticRegression()


class SVCWrapper(SklearnClassificationWrapper):
    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.model = sklearn.svm.SVC(probability=True)