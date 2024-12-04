import os

import hydra
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf

import rootutils

rootutils.setup_root(__file__, pythonpath=True)

import gradio as gr
# Inputs => Dropdown populated with list of examples from train set + text box for custom input
# Outputs => Prediction + ground truth, bars for probability
# fn => predict function
from multivariate.data import MODTimeseries, MODTimeseriesModule

# Iterate through all saved models and evaluate them on the test set.
# Load the data
@hydra.main(config_path=os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "conf"),
            config_name="config",
            version_base="1.1")
def main(config: DictConfig) -> None:
    dm = MODTimeseriesModule(config, reset_cache=True)
    dm.setup()

    # Get list of examples from train set
    train_indices = dm.train_indices
    example_dropdown_choices = [f"{i}" for i in train_indices]
    example_dropdown_values = train_indices

    with (gr.Blocks() as demo):
        with gr.Row():
            with gr.Column():
                example_dropdown = gr.Dropdown(label="Patient", choices=example_dropdown_choices, type="index")
                inputs = [example_dropdown]

            with gr.Column():
                status = gr.Text()
                motion_plot = gr.Plot(label="Log Motion")
                light_plot = gr.Plot(label="Log Luminosity")
                outputs = [motion_plot, light_plot, status]

            def update_lineplot(*input):
                # Get the example.
                example = dm.dataset.load_example(example_dropdown_choices[input[0]])
                status_out = f"Birth Status: {example['birth_status']}\n" \
                             f"Actual - Estimated Diff: {example['actual_diff']}\n" \
                             f"Examples: {example['examples_len']}\n"
                # Generate Series objects for each example.
                # dfs = []
                # for date, data in example["examples"].items():
                #     df = pd.DataFrame({
                #         "motion": data["motion"],
                #         "light": data["light"],
                #         "time": data["time"],
                #         "days_before_actual": data["days_before_actual"],
                #     })
                #     dfs.append(df)
                light_series = []
                motion_series = []
                for date, data in example["examples"].items():
                    light_series.append(pd.Series(data["light"], index=data["time"], name=data["days_before_estimated"]))
                    motion_series.append(pd.Series(data["motion"], index=data["time"], name=data["days_before_estimated"]))

                # Combine into DataFrames
                light_df = pd.concat(light_series, axis=1)
                motion_df = pd.concat(motion_series, axis=1)
                def plot_df(df, type, transform):
                    # Plot each column as a row in a grid x = time, y = col name, color = value
                    fig, ax = plt.subplots()
                    vmin = transform(config["data"]["raw_config"][type]["min"])
                    vmax = transform(config["data"]["raw_config"][type]["max"])
                    ax.imshow(transform(df.values.T), cmap="gray", aspect="auto", interpolation="none",
                              vmin=vmin, vmax=vmax)
                    # Set x-axis to index values, spaced out to every 12 hours (60 minutes)
                    t = np.arange(len(df.index), step=12 * 60)
                    t = np.append(t, len(df.index) - 1)
                    ax.set_xticks(t)
                    ax.set_xticklabels(df.index[t])
                    # Set y-axis to column names, only showing where there are discontinuities above 1 day
                    days = df.columns.values
                    days_to_show = []
                    for i in range(len(days)):
                        if i == 0:
                            days_to_show.append(i)
                        else:
                            if np.abs(days[i] - days[i - 1]) > 1:
                                days_to_show.append(i - 1)
                                days_to_show.append(i)
                    days_to_show.append(len(days) - 1)
                    ax.set_yticks(days_to_show)
                    ax.set_yticklabels(days[days_to_show])
                    ax.set_xlabel("Minutes")
                    ax.set_ylabel("Days Before EDC")
                    return fig
                return plot_df(motion_df,"motion", lambda x: x), plot_df(light_df, "light", lambda x: x), status_out

            example_dropdown.select(update_lineplot, example_dropdown, outputs)


    demo.launch()

if __name__ == "__main__":
    main()
