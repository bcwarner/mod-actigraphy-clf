import os

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

import rootutils

from baseline.sklearn_models import SklearnWrapper, SklearnClassificationWrapper
import shap

rootutils.setup_root(__file__, pythonpath=True)

import gradio as gr
# Inputs => Dropdown populated with list of examples from train set + text box for custom input
# Outputs => Prediction + ground truth, bars for probability
# fn => predict function
from baseline.data import MODTabularDataModule, MODTabularDataset

# Iterate through all saved models and evaluate them on the test set.
# Load the data
@hydra.main(config_path=os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "conf"),
            config_name="config",
            version_base="1.1")
def main(config: DictConfig) -> None:
    dm = MODTabularDataModule(config)
    dm.setup()

    # Get list of examples from train set
    train_indices = dm.train_indices
    example_dropdown_choices = [f"{dm.dataset.indices_to_patient_id[i]} ({i})" for i in train_indices]
    example_dropdown_values = train_indices

    # Get list of models from saved models directory
    saved_models = os.listdir(config["models"]["saved"])
    models = []
    labels = {}
    for model_name in saved_models:
        model_path = os.path.normpath(os.path.join(config["models"]["saved"], model_name))
        model = SklearnWrapper.load(model_path, config=config)
        models.append(model)
        labels[len(models)] = model.target_label

    model_dropdown_choices = [f"{model.__class__.__name__} ({i})" for i, model in enumerate(models)]

    # Pull a list of features from the dataset
    features = list(dm.dataset.feature_names)

    def predict(*input):
        # Get the model
        model = models[input[0]]
        # Get the prediction
        fts = pd.DataFrame([{k: v for k, v in zip(features, input[1:])}])
        # Get the probability
        if hasattr(model, "predict_proba"):
            probability = {k: v for k, v in enumerate(model.predict_proba(fts)[0])}
        else:
            probability = {}
        prediction = model.predict(fts)
        return prediction, probability

    def interpret(*input):
        # Get the model
        model = models[input[0]]
        # Get the prediction
        fts = pd.DataFrame([{k: v for k, v in zip(features, input[1:])}])
        # Get the probability
        explainer = shap.Explainer(model)
        scores = explainer(fts, seed=config["seed"])
        plot = shap.plots.bar(scores)
        return plot

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                inputs = [
                          gr.Dropdown(label="Model", choices=model_dropdown_choices, type="index")]

                feature_inputs = {}
                for feature in features:
                    feature_inputs[feature] = gr.Number(label=feature)
                    inputs.append(feature_inputs[feature])

                pred = gr.Button(value="Predict")
                explain = gr.Button(value="Explain")

            with gr.Column():
                gt = gr.Textbox(label=models[0].target_label)
                outputs = [gr.Textbox(label="Prediction"),
                           gr.Label(label="Probability")]
                shap_plot = gr.Plot(label="SHAP Plot")

                pred.click(predict, inputs=inputs, outputs=outputs)
                explain.click(interpret, inputs=inputs, outputs=shap_plot)

        with gr.Row():
            serialized_examples = []

            for features, label in dm.train_dataloader():
                serialized_examples.append([x.item() for x in features.values()] +
                                           [label[models[0].target_label].item(),
                                            ])
                if len(serialized_examples) > 100:
                    break

            gr.Examples(
                examples=serialized_examples,
                inputs=list(feature_inputs.values()) + [gt],
            )

    demo.launch()

if __name__ == "__main__":
    main()
