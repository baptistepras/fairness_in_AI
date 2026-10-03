# Fairness in AI

Measures and reduces the bias of a chest X-ray classifier that predicts whether a patient is sick. Bias is measured with the true and false positive rates of each group (age, sex, and both), and two families of methods try to reduce it:

* **Pre-processing**: reweighting the training samples by group, by group and label, or with the Kamiran and Calders method;
* **Post-processing**: reject option classification and equalized odds, applied to the predictions of a trained model.

Each method is evaluated with the balanced and standard accuracies and the gaps between groups. The plots in `plots/` compare the rates before and after each method, and `RAPPORT.pdf` (in French) presents the full study.

## Data

`selected_data/` holds a subset of the NIH ChestX-ray14 dataset (1,125 training and 375 validation images) and its metadata. The dataset is provided by the NIH Clinical Center (https://nihcc.app.box.com/v/ChestXray-NIHCC) and described in:

> X. Wang, Y. Peng, L. Lu, Z. Lu, M. Bagheri, and R. M. Summers. ChestX-ray8: Hospital-Scale Chest X-Ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases. *CVPR*, 2017.

## Usage

```bash
conda env create -f fairness_environment.yml
```

* `train_classifier()` in `train_classifieur.ipynb` trains a model and saves a checkpoint in `expe_log/` (about 15 minutes).
* `pred_classifier()` evaluates a checkpoint (set `ckpt_path`) and writes `expe_log/preds.csv` (about 5 minutes).
* `main.py` runs the analysis: pre-processing methods update the sample weights in `metadata.csv` before training, and post-processing methods adjust `preds.csv`.

Checkpoints weigh over 100 MB and are not included.

## Credits

`train_classifieur.py` and `train_classifieur.ipynb` were provided as starter code. The bias analysis and mitigation methods are in `main.py`.

## License

MIT, see [LICENSE](LICENSE). The starter code (`train_classifieur.py`, `train_classifieur.ipynb`) and the NIH images are not covered.
