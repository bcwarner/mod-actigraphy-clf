# Plots the evaluation results from baseline/evaluate.py

import argparse
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
import textwrap
from itertools import permutations
from typing import Dict, List

import lightning.pytorch as pl
import numpy as np
import scipy
import shap
import torch
import yaml
from lightning import Callback
from lightning.pytorch.callbacks import TQDMProgressBar, EarlyStopping
from lightning.pytorch.profilers import PyTorchProfiler, AdvancedProfiler
import torchmetrics
import pickle

from data import MODTabularDataModule, MODTabularDataset
from sklearn_models import *
from torcheval.metrics.aggregation.auc import AUC

import hydra
from omegaconf import DictConfig, OmegaConf
from evaluate import EvaluationResult

import matplotlib.pyplot as plt

import rootutils
rootutils.setup_root(__file__, pythonpath=True)

from baseline.custom_shap_plots import bar_custom
import matplotlib.patches as mpatches

def map_name(config, val):
    if val not in config["names"]:
        return val
    return config["names"][val]

def plot_classification(config: DictConfig,
                                 prob_metrics_comparison: Dict[str, EvaluationResult]):

    model_metrics = defaultdict(dict) # Model => metric => value

    # List of unique ablation configs
    ablation_configs = set([result.ablation_config for result in prob_metrics_comparison.values()])
    table_results = []
    for ablation_config in ablation_configs:
        plt.clf()
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        auroc_ax = ax[0]
        auprc_ax = ax[1]

        # AUROC curve
        for model_fpath, results in prob_metrics_comparison.items():
            if results.ablation_config != ablation_config:
                continue # Wacky
            roc_metric: torchmetrics.Metric = torchmetrics.ROC(task="binary")
            spec_at_sens = torchmetrics.SpecificityAtSensitivity(task="binary", min_sensitivity=config.eval.spec_at_sens)
            roc_metric.update(results.y_prob.type(torch.float32), results.y.type(torch.int))
            spec_at_sens.update(results.y_prob.type(torch.float32), results.y.type(torch.int))
            fpr, tpr, thresholds = roc_metric.compute()
            auroc = AUC()
            auroc.update(fpr, tpr)
            auroc = auroc.compute().item()
            model_name_nice = map_name(config, results.model_name)
            next_auroc_result = {
                "model": results.model_name,
                "ablation_config": ablation_config,
                "auroc": auroc,
                "spec_at_sens": spec_at_sens.compute()[0].item(),
            }
            table_results.append(next_auroc_result)
            auroc_ax.plot(fpr, tpr, label=f"{model_name_nice} (AUC = {auroc:.2f})")
        auroc_ax.set_xlabel("False Positive Rate")
        auroc_ax.set_ylabel("True Positive Rate")
        auroc_ax.legend()
        auroc_ax.set_title("\\textbf{a)} ROC Curves")

        # AUPRC curve
        for model_fpath, results in prob_metrics_comparison.items():
            if results.ablation_config != ablation_config:
                continue
            prc_metric: torchmetrics.Metric = torchmetrics.PrecisionRecallCurve(task="binary")
            prc_metric.update(results.y_prob.type(torch.float32), results.y.type(torch.int))
            precision, recall, thresholds = prc_metric.compute()
            auprc = AUC()
            auprc.update(recall, precision)
            auprc = auprc.compute().item()
            model_name_nice = map_name(config, results.model_name)
            next_auprc_result = {
                "model": results.model_name,
                "ablation_config": ablation_config,
                "auprc": auprc,
            }
            table_results.append(next_auprc_result)
            auprc_ax.plot(recall, precision, label=f"{model_name_nice} (AUC={auprc:.2f})")


        auprc_ax.set_xlabel("Recall")
        auprc_ax.set_ylabel("Precision")
        auprc_ax.legend()
        auprc_ax.set_title("\\textbf{b)} PRC Curves")
        plt.suptitle(f"ROC and PRC Curves ({map_name(config, ablation_config)})")
        plt.tight_layout()
        plt.savefig(os.path.normpath(os.path.join(config["models"]["plots"], f"auroc_auprc_curve_{ablation_config}.pdf")))

        # Join the table results by the model/ablation config to merge the metrics
        table_df = pd.DataFrame(table_results)
        table_df = table_df.groupby(["model", "ablation_config"]).last().reset_index()

        # Remap model names to nice names
        table_df["model"] = table_df["model"].apply(lambda x: map_name(config, x))
        # Remap the values to nice names
        table_df["ablation_config"] = table_df["ablation_config"].apply(lambda x: map_name(config, x))
        # Remap the column names to nice names
        table_df = table_df.rename(columns={k: map_name(config, k) for k in table_df.columns})

        indices = [map_name(config, "model"), map_name(config, "ablation_config")]
        for new_index in permutations(indices):
            table_df_new = table_df.copy()
            # Trim to 3 decimal places
            table_df_new = table_df_new.round(3)
            # Sort by the main column
            table_df_new = table_df_new.sort_values(by=new_index)
            table_df_new.to_csv(os.path.normpath(os.path.join(config["models"]["plots"], f"classification_metrics_{new_index[0]}.csv")))
            # Save as a LaTeX table
            table_df_new.to_latex(os.path.normpath(os.path.join(config["models"]["plots"], f"classification_metrics_{new_index[0]}.tex")),
                                  index=False, escape=False, bold_rows=True, multicolumn_format="c", multirow=True)

def plot_classification_multiple(config: DictConfig,
                                 prob_metrics_comparison: Dict[str, Dict[str, List[EvaluationResult]]]):

    model_metrics = defaultdict(dict) # Model => metric => value
    # List of unique ablation configs
    table_results = []
    results_by_model_ablation = defaultdict(list)
    ablation_plots = defaultdict(dict) # Ablation config => model => metrics
    for model_fpath, models in prob_metrics_comparison.items():
        for ablations, runs in models.items():
            plt.clf()
            # AUROC curve
            run_fpr, run_tpr = [], []
            run_auroc = []
            run_precision, run_recall = [], []
            run_auprc = []
            run_spec_at_sens = []
            pooled_roc_metric: torchmetrics.Metric = torchmetrics.ROC(task="binary")#, thresholds=thresholds)
            pooled_prc_metric: torchmetrics.Metric = torchmetrics.PrecisionRecallCurve(task="binary")#, thresholds=thresholds)
            for results in runs:
                y_prob = results.y_prob.type(torch.float32)
                y = results.y.type(torch.int).clip(0, 1)

                pooled_roc_metric.update(y_prob, y)
                pooled_prc_metric.update(y_prob, y)

                roc_metric: torchmetrics.Metric = torchmetrics.ROC(task="binary")
                roc_metric.update(y_prob, y)
                fpr, tpr, thresholds = roc_metric.compute()

                spec_at_sens = torchmetrics.SpecificityAtSensitivity(task="binary", min_sensitivity=config.eval.spec_at_sens)
                spec_at_sens.update(y_prob, y)

                auroc = AUC()

                auroc.update(fpr, tpr)
                run_fpr.append(fpr)
                run_tpr.append(tpr)
                auroc = auroc.compute().item()
                run_auroc.append(auroc)

                prc_metric: torchmetrics.Metric = torchmetrics.PrecisionRecallCurve(task="binary")
                prc_metric.update(y_prob, y)
                precision, recall, thresholds = prc_metric.compute()
                run_recall.append(recall)
                run_precision.append(precision)

                auprc = AUC()
                auprc.update(recall, precision)
                auprc = auprc.compute().item()
                run_auprc.append(auprc)

                run_spec_at_sens.append(spec_at_sens.compute()[0].item())

            fpr_pooled, tpr_pooled, _ = pooled_roc_metric.compute()
            precision_pooled, recall_pooled, _ = pooled_prc_metric.compute()
            auroc_p = AUC()
            auroc_p.update(fpr_pooled, tpr_pooled)
            auroc_pooled = auroc_p.compute().item()
            auprc_p = AUC()
            auprc_p.update(recall_pooled, precision_pooled)
            auprc_pooled = auprc_p.compute().item()

            ablation_plots[ablations][map_name(config, model_fpath)] = {"fpr_mean": fpr_pooled,
                                                                        "tpr_mean": tpr_pooled,
                                                                        "auroc_mean": auroc_pooled,
                                                                        "auprc_mean": auprc_pooled,
                                                                        "precision_mean": precision_pooled,
                                                                        "recall_mean": recall_pooled,}

            auroc_mean = np.mean(run_auroc)
            auroc_std = np.std(run_auroc)
            auroc_ci = scipy.stats.t.interval(0.95, len(run_auroc) - 1, loc=auroc_mean, scale=scipy.stats.sem(run_auroc))
            auroc_ci_latex = f"{auroc_mean:.3f} ({auroc_ci[0]:.3f} - {auroc_ci[1]:.3f})"
            auprc_mean = np.mean(run_auprc)
            auprc_std = np.std(run_auprc)
            auprc_ci = scipy.stats.t.interval(0.95, len(run_auprc) - 1, loc=auprc_mean, scale=scipy.stats.sem(run_auprc))
            auprc_ci_latex = f"{auprc_mean:.3f} ({auprc_ci[0]:.3f} - {auprc_ci[1]:.3f})"
            spec_at_sens_mean = np.mean(run_spec_at_sens)
            spec_at_sens_std = np.std(run_spec_at_sens)
            spec_at_sens_ci = scipy.stats.t.interval(0.95, len(run_spec_at_sens) - 1, loc=spec_at_sens_mean, scale=scipy.stats.sem(run_spec_at_sens))
            spec_at_sens_ci_latex = f"{spec_at_sens_mean:.3f} ({spec_at_sens_ci[0]:.3f} - {spec_at_sens_ci[1]:.3f})"


            table_results.append({
                "model": model_fpath,
                "ablation_config": ablations,
                "auroc_pooled": auroc_pooled,
                #"auroc_mean": auroc_mean,
                #"auroc_std": auroc_std,
                "auroc_ci": auroc_ci_latex,
                #"auprc_mean": auprc_mean,
                #"auprc_std": auprc_std,
                "auprc_pooled": auprc_pooled,
                "auprc_ci": auprc_ci_latex,
                #"spec_at_sens_mean": spec_at_sens_mean,
                #"spec_at_sens_std": spec_at_sens_std,
                "spec_at_sens_ci": spec_at_sens_ci_latex,
            })
            results_by_model_ablation[ablations].append({
                "model": model_fpath,
                "ablation_config": ablations,
                "auroc_mean": auroc_mean,
                "auroc_std": auroc_std,
                "auroc_ci": auroc_ci_latex,
                "auprc_mean": auprc_mean,
                "auprc_std": auprc_std,
                "auprc_ci": auprc_ci_latex,
                "spec_at_sens_mean": spec_at_sens_mean,
                "spec_at_sens_std": spec_at_sens_std,
                "spec_at_sens_ci": spec_at_sens_ci_latex,
            })

    table_df = pd.DataFrame(table_results)
    # Remap model names to nice names
    table_df["model"] = table_df["model"].apply(lambda x: map_name(config, x))
    # Remap the values to nice names
    table_df["ablation_config"] = table_df["ablation_config"].apply(lambda x: map_name(config, x))
    # Remap the column names to nice names
    table_df = table_df.rename(columns={k: map_name(config, k) for k in table_df.columns})

    indices = [map_name(config, "model"), map_name(config, "ablation_config")]
    for new_index in permutations(indices):
        table_df_new = table_df.copy()
        # Trim to 3 decimal places
        table_df_new = table_df_new.round(3)
        # Sort by the main column
        table_df_new = table_df_new.sort_values(by=new_index[1])
        table_df_new = table_df_new.sort_values(by=new_index[0], kind="mergesort")
        table_df_new.to_csv(os.path.normpath(
            os.path.join(config["models"]["plots"], f"multiple_classification_metrics_{new_index}.csv")))
        # Save as a LaTeX table
        table_df_new.to_latex(os.path.normpath(
            os.path.join(config["models"]["plots"], f"multiple_classification_metrics_{new_index}.tex")),
                              index=False, escape=False, bold_rows=True, multicolumn_format="c", multirow=True)

    for ablation in ablation_plots:
        plt.clf()
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        auroc_ax = ax[0]
        auprc_ax = ax[1]
        for model, metrics in ablation_plots[ablation].items():
            auroc_ax.plot(metrics["fpr_mean"], metrics["tpr_mean"], label=f"{model} (AUC = {metrics['auroc_mean']:.2f})")
        auroc_ax.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), color="black", linestyle="--")
        auroc_ax.set_title("\\textbf{a)} Pooled ROC Curve")
        auroc_ax.set_xlabel("False Positive Rate")
        auroc_ax.set_ylabel("True Positive Rate")
        auroc_ax.legend(fontsize="small")

        for model, metrics in ablation_plots[ablation].items():
            auprc_ax.plot(metrics["recall_mean"], metrics["precision_mean"], label=f"{model} (AUC = {metrics['auprc_mean']:.2f})")
        auprc_ax.set_title("\\textbf{b)} Pooled PRC Curve")
        auprc_ax.set_xlabel("Recall")
        auprc_ax.set_ylabel("Precision")
        auprc_ax.legend(fontsize="small")
        plt.suptitle(f"Pooled ROC and PRC Curves ({map_name(config, ablation)})")
        plt.savefig(os.path.normpath(os.path.join(config["models"]["plots"], f"multiple_auprc_auroc_curve_{ablation}.pdf")))




def plot_regression(config, reg_metrics_comparison):
    model_metrics = []

    for model_fpath, results in reg_metrics_comparison.items():
        metrics = torchmetrics.MetricCollection([torchmetrics.MeanSquaredError(),
                                                torchmetrics.MeanAbsoluteError(),
                                                torchmetrics.R2Score(),
                                                 torchmetrics.MeanAbsolutePercentageError()])

        metrics_results = metrics(results.y_pred, results.y)
        metrics_results = {k: v.item() for k, v in metrics_results.items()}

        model_metrics.append({"model": results.model_name,
                              **metrics_results})

        plt.clf()
        # Scatter plot of predictions vs ground truth
        fig, ax = plt.subplots()
        ax.scatter(results.y, results.y_pred, s=3)
        ax.set_xlim(0, 40)
        ax.set_ylim(0, 40)
        ax.plot(np.linspace(0, 40, 100), np.linspace(0, 40, 100), color="black", linestyle="--")
        ax.set_xlabel("Ground Truth")
        ax.set_ylabel("Prediction")
        ax.set_title(f"{results.model_name} ({results.ablation_config}) Predictions vs Ground Truth")
        plt.savefig(os.path.normpath(os.path.join(config["models"]["plots"], f"{results.model_name}_scatter.pdf")))

    df = pd.DataFrame(model_metrics)
    df.to_csv(os.path.normpath(os.path.join(config["models"]["plots"], "regression_metrics.csv")))


def plot_shap(config, metrics_comparison):
    redcap_fnames = pd.read_csv(config["data"]["redcap_feature_names"]["file"])
    redcap_fnames[config["data"]["redcap_feature_names"]["section_name"]] = redcap_fnames[config["data"]["redcap_feature_names"]["section_name"]].ffill()

    green = np.array([0.20784314, 0.58039216, 0.07843137])
    gold = np.array([0.58823529, 0.49803922, 0])

    def feature_name_color_mapper_all(fname):
        if "std" in fname or "mean" in fname:
            return shap.plots.colors.blue_rgb, "x"
        else:
            # Return green for socio-economic features
            if fname in ['age_enroll', 'education', 'employed', 'ethnicity', 'income_annual1', 'marital', 'race']:
                return green, "/"
            else:
                return shap.plots.colors.red_rgb, "\\"
            
    feature_name_color_patches = [mpatches.Patch(color=shap.plots.colors.blue_rgb, label="Actigraphy", hatch="x"),
                                  mpatches.Patch(color=green, label="SES", hatch="/"),
                                  mpatches.Patch(color=shap.plots.colors.red_rgb, label="Other CRF", hatch="\\")]

    feature_name_color_patches_actigraphy = [mpatches.Patch(color=shap.plots.colors.blue_rgb, label="Std. Dev.", hatch="x"),
                                            mpatches.Patch(color=gold, label="Avg.", hatch="/"),]
    
    def feature_name_color_mapper_actigraphy_only(fname):
        if "std" in fname:
            return shap.plots.colors.blue_rgb, "x"
        else:
            return gold, "/"

    
    def feature_name_mapper(fname):
        if fname.replace("_mean", "").replace("_std", "") in config["names"]:
            base_name = config["names"][fname.replace("_mean", "").replace("_std", "")]
            if "std" in fname:
                return f"{base_name} (Std. Dev.)"
            elif "mean" in fname:
                return f"{base_name} (Avg.)"
            else: 
                return base_name
        elif fname in redcap_fnames[config["data"]["redcap_feature_names"]["feature_name"]].unique():
            feature = redcap_fnames.loc[redcap_fnames[config["data"]["redcap_feature_names"]["feature_name"]] == fname]
            fname_new = feature[config["data"]["redcap_feature_names"]["field_label"]].item()
            if len(fname_new.strip().split()) == 1:
                section_name = feature[config["data"]["redcap_feature_names"]["section_name"]].item()
                if not pd.isna(section_name):
                    fname_new = section_name + ": " + fname_new
            return fname_new
        
        fname = "\n".join(textwrap.wrap(fname, 20))

        return fname


    for model, results in metrics_comparison.items():
        # Beeswarm plot
        plt.clf()
        sv = results.shap_values
        if len(sv.values.shape) > 2:
            sv = sv[:, :, 1]
        nice_name_title = f"{map_name(config, results.model_name)} ({map_name(config, results.ablation_config)})"
        save_name = f"{model.replace('Wrapper.pkl', '')}_{results.ablation_config}"
        max_shap = sv.shape[1] if config["plot"]["max_shap_feats"] is None else config["plot"]["max_shap_feats"]
        # shap.plots.beeswarm(sv, show=False, max_display=max_shap)
        #plt.title(f"{nice_name_title} SHAP Values")
        ## Widen the plot
        #plt.gcf().set_size_inches(8.5, 11)
        #plt.gcf().tight_layout()
        #plt.savefig(os.path.normpath(os.path.join(config["models"]["plots"], f"{save_name}_shap_beeswarm.pdf")), bbox_inches="tight")
        #plt.clf()

        color_mapper = feature_name_color_mapper_all if not ("no_sts" in results.ablation_config) else feature_name_color_mapper_actigraphy_only
        color_handles = feature_name_color_patches if not ("no_sts" in results.ablation_config) else feature_name_color_patches_actigraphy

        # Summary plot
        bar_custom(sv, show=False, max_display=max_shap, feature_name_color_mapper=color_mapper, feature_names_mapper=feature_name_mapper)
        plt.suptitle(f"{nice_name_title} SHAP Values")
        plt.legend(handles=color_handles, loc="lower right", fontsize="small")
        plt.gcf().set_size_inches(8.5, 11)
        plt.gcf().tight_layout()
        plt.savefig(os.path.normpath(os.path.join(config["models"]["plots"], f"{save_name}_shap_bar.pdf")), bbox_inches="tight")

def plot_shap_multiple(config, prob_metrics_comparison):
    model_metrics = defaultdict(dict) # Model => metric => value
    # List of unique ablation configs

    redcap_fnames = pd.read_csv(config["data"]["redcap_feature_names"]["file"])
    redcap_fnames[config["data"]["redcap_feature_names"]["section_name"]] = redcap_fnames[config["data"]["redcap_feature_names"]["section_name"]].ffill()

    green = np.array([0.20784314, 0.58039216, 0.07843137])
    gold = np.array([0.58823529, 0.49803922, 0])

    def feature_name_color_mapper_all(fname):
        if "std" in fname or "mean" in fname:
            return shap.plots.colors.blue_rgb, "x"
        else:
            # Return green for socio-economic features
            if fname in ['age_enroll', 'education', 'employed', 'ethnicity', 'income_annual1', 'marital', 'race']:
                return green, "/"
            else:
                return shap.plots.colors.red_rgb, "\\"
            
    feature_name_color_patches = [mpatches.Patch(color=shap.plots.colors.blue_rgb, label="Actigraphy", hatch="x"),
                                  mpatches.Patch(color=green, label="SES", hatch="/"),
                                  mpatches.Patch(color=shap.plots.colors.red_rgb, label="Other CRF", hatch="\\")]

    feature_name_color_patches_actigraphy = [mpatches.Patch(color=shap.plots.colors.blue_rgb, label="Std. Dev.", hatch="x"),
                                            mpatches.Patch(color=gold, label="Avg.", hatch="/"),]


    def feature_name_color_mapper_actigraphy_only(fname):
        if "_std" in fname:
            return shap.plots.colors.blue_rgb, "x"
        else:
            return gold, "/"

    def feature_name_mapper(fname):
        if fname.replace("_mean", "").replace("_std", "") in config["names"]:
            base_name = config["names"][fname.replace("_mean", "").replace("_std", "")]
            if "std" in fname:
                return f"{base_name} (Std. Dev.)"
            elif "mean" in fname:
                return f"{base_name} (Avg.)"
            else: 
                return base_name
        elif fname in redcap_fnames[config["data"]["redcap_feature_names"]["feature_name"]].unique():
            feature = redcap_fnames.loc[redcap_fnames[config["data"]["redcap_feature_names"]["feature_name"]] == fname]
            fname_new = feature[config["data"]["redcap_feature_names"]["field_label"]].item()
            if len(fname_new.strip().split()) == 1:
                section_name = feature[config["data"]["redcap_feature_names"]["section_name"]].item()
                if not pd.isna(section_name):
                    fname_new = section_name + ": " + fname_new
            return fname_new
        
        fname = "\n".join(textwrap.wrap(fname, 20))

        return fname

    results_by_model_ablation = defaultdict(list)
    for model_fpath, models in prob_metrics_comparison.items():
        for ablations, runs in models.items():
            plt.clf()
            for results in runs:
                results_by_model_ablation[(model_fpath, ablations)].append(results)

    # Average the SHAP values together
    for ablation_config, results in results_by_model_ablation.items():
        plt.clf()
        shap_values = [result.shap_values.abs for result in results]
        shap_values = sum(shap_values) / len(shap_values)
        if len(shap_values.shape) > 2:
            shap_values = shap_values[:, :, 1]


        nice_name_title = f"{map_name(config, results[0].model_name)} ({map_name(config, results[0].ablation_config)})"
        max_shap = shap_values.shape[1] if config["plot"]["max_shap_feats"] is None else config["plot"]["max_shap_feats"]
        save_name = f"{'_'.join(ablation_config)}_multiple"
        actigraphy_only = "no_redcap" not in results[0].ablation_config and "no_sts" not in  results[0].ablation_config
        color_mapper = feature_name_color_mapper_all if actigraphy_only else feature_name_color_mapper_actigraphy_only
        color_handles = feature_name_color_patches if actigraphy_only else feature_name_color_patches_actigraphy

        # Summary plot
        bar_custom(shap_values, show=False, max_display=max_shap, feature_name_color_mapper=color_mapper, feature_names_mapper=feature_name_mapper)
        plt.suptitle(f"{nice_name_title} SHAP Values")
        plt.legend(handles=color_handles, loc="lower right", fontsize="small")
        plt.gcf().set_size_inches(8.5, 11)
        plt.gcf().tight_layout()
        plt.savefig(os.path.normpath(os.path.join(config["models"]["plots"], f"{save_name}_shap_bar.pdf")), bbox_inches="tight")

def plot_horizon(config, metrics_comparison):
    output_df_rows = []
    for results in metrics_comparison:
        if results.seed != config["seed"]:
            continue
        roc_metric: torchmetrics.Metric = torchmetrics.ROC(task="binary")
        prc_metric: torchmetrics.Metric = torchmetrics.PrecisionRecallCurve(task="binary")
        y_prob = results.y_prob.type(torch.float32)
        y = results.y.type(torch.int).clip(0, 1)  # Filter out the PTB++ patient
        roc_metric.update(y_prob, y)
        prc_metric.update(y_prob, y)
        fpr, tpr, thresholds = roc_metric.compute()
        precision, recall, thresholds = prc_metric.compute()
        auroc = AUC()
        auprc = AUC()
        auroc.update(fpr, tpr)
        auprc.update(recall, precision)
        auroc = auroc.compute().item()
        auprc = auprc.compute().item()

        next_auroc_result = {
            "model": results.model_name,
            "horizon": results.horizon,
            "auroc": auroc,
            "auprc": auprc,
            "y_count": len(y),
            "ablation_config": results.ablation_config,
            "seed": results.seed,
        }
        output_df_rows.append(next_auroc_result)

    output_df = pd.DataFrame(output_df_rows)
    output_df.to_csv(os.path.normpath(os.path.join(config["models"]["plots"], "horizon_metrics.csv")))
    # Plot by the horizon, grouping by the model
    # Initialize two figures
    plt.clf()
    fig, ax = plt.subplots(2, 1, figsize=(7, 7))
    ax_auroc = ax[0]
    ax_auprc = ax[1]

    ablation_config = output_df["ablation_config"].value_counts().idxmax()

    for model in output_df["model"].unique():
        model_df = output_df[output_df["model"] == model]
        if len(model_df) < 2:
            continue
        # Select the most common ablation config
        model_df = model_df[model_df["ablation_config"] == ablation_config]

        # Sort by the horizon
        model_df = model_df.sort_values(by="horizon")

        ax_auroc.plot(model_df["horizon"], model_df["auroc"])#, label=map_name(config, model))
        ax_auprc.plot(model_df["horizon"], model_df["auprc"], label=map_name(config, model))

    # Set subplot titles
    ax_auroc.set_title("\\textbf{a)} AUROC")
    xlabel = "Maximum Gestational Age (Days)"
    ax_auroc.set_xlabel(xlabel)
    ax_auroc.set_ylabel("AUROC")

    ax_auprc.set_title("\\textbf{b)} AUPRC")
    ax_auprc.set_xlabel(xlabel)
    ax_auprc.set_ylabel("AUPRC")

    fig.suptitle(f"AUROC and AUPRC By Horizon ({map_name(config, ablation_config)})")
    ax_auprc.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.normpath(os.path.join(config["models"]["plots"], f"horizon_metrics.pdf")))

@hydra.main(config_path=os.path.join(os.path.dirname(os.getcwd()), "conf"),
            config_name="config",
            version_base="1.1")
def main(config: DictConfig) -> None:
    evaluation_path = os.path.normpath(os.path.join(config["models"]["results"]))

    class_metrics_comparison = defaultdict(EvaluationResult)
    multirun_metrics_comparison = defaultdict(lambda: defaultdict(list))
    class_horizon_metrics_comparison = []
    reg_metrics_comparison = defaultdict(EvaluationResult)

    plt.rc("text", usetex=True)

    for evaluation_file in os.listdir(evaluation_path):
        evaluation_file_path = os.path.normpath(os.path.join(evaluation_path, evaluation_file))
        if evaluation_file[0] == ".": # Skip hidden files
            continue

        with open(evaluation_file_path, "rb") as f:
            evaluation_result = pickle.load(f)

        model_name = evaluation_file_path.split("/")[-1].replace(".pkl", "")
        if len(evaluation_result.y_prob) > 0: # Classification
            if "horizon" in dir(evaluation_result):
                class_horizon_metrics_comparison.append(evaluation_result)
            if "seed" in dir(evaluation_result) and evaluation_result.seed != config["seed"]:
                multirun_metrics_comparison[evaluation_result.model_name][evaluation_result.ablation_config].append(evaluation_result)
            if ("horizon" not in dir(evaluation_result) or evaluation_result.horizon == config["horizon"]) \
                and ("seed" not in dir(evaluation_result) or evaluation_result.seed == config["seed"]):
                class_metrics_comparison[model_name] = evaluation_result
        else: # Regression
            reg_metrics_comparison[model_name] = evaluation_result

    # Plot the metrics
    #plot_classification(config, class_metrics_comparison)
    plot_classification_multiple(config, multirun_metrics_comparison)
    #plot_regression(config, reg_metrics_comparison)
    plot_horizon(config, class_horizon_metrics_comparison)
    #plot_shap(config, class_metrics_comparison)
    plot_shap_multiple(config, multirun_metrics_comparison)


if __name__ == "__main__":
    main()