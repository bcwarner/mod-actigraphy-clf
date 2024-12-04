# Validation of sleep-based actigraphy machine learning models for prediction of preterm birth

This is the code for our paper, ``Validation of sleep-based actigraphy machine learning models for prediction of preterm birth`` (in submission).

## Usage

Requirements are outlined in `requirements.txt` and can be installed via `pip install -r requirements.txt`. Data is not included, but if you wish to run your own survey data and/or summarized actigraphy data, references can be updated in the `conf/conf.yaml` file. 

This project relies on the Hydra configuration system, with `conf/conf.yaml` containing the majority of the configuration. `conf/ablation`, `conf/model` contain the configurations for the ablation studies and model training, respectively. `conf/names` is used to map internal names to readable names. You will need to add a `paths/root.yaml` file pointing to the base of your data directory.

Training is done with `python train.py`, here are several example commands: 

```
python train.py model=LinearSVCClassifier # Run linear SVC, base seed, all features
python train.py -m seed=175,4325,5132,09645,10239,23425,024248,59875,62345,657095 ablation=all_fts,no_sts,no_tabular model=glob(*) # All main models, with ablation studies
python train.py -m model=glob(*) ablation=all_nulliparous # One seed, all models, all nulliparous patients
```

Evaluation can then be done with `python evaluate_all.py` and plotting `python plot.py`.
