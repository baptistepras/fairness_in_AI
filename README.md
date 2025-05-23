# Fairness in AI

This project implements different pre-processing and post-processing to reduce bias in model predictions on a dataset where the goal is to define if a patient is sick or not, using a chest X-ray.

The only code made by me is in the file `main.py`, the other `.py` and `.ipynb` files were given to me as a starter for the project.

# How to use it ?

Download or replicate the environment `fairness-environment.yml`.

Using `train_classifier()` in `train_classifier.ipynb()` creates a new model, saved in a .ckpt extension in the repository `expe_log`.
Using `pred_classifier()` in `train_classifier.ipynb()` evaluates a model and creates `preds.csv` in the repository `expe_log`. Use `ckpt_path` to specify which model you want to use.

Use `main.py` to use different pre-processing and post-processing methods. Pre-processing methods modify the weights in the file `metadata.csv`, which are then used when you train a model. Post-processing methods use the file `preds.csv` to adjust the predictions and then re-evaluate the performances and metrics. 

The metrics used are the `True Positive Rate` and the `False Positive Rate` for the different categories concerned by a bias, and the accuracies given are the `balanced accuracy` and the `classic accuracy`.

# Warnings

The models saved under a `ckpt` extension are quite heavy (+100Mo) and are therefore not available here, but the full results of each model are available in the `.txt` files and can be easily seen with the plots in the repository `plots`.

`train_classifier()` can take up to 15 minutes and `pred_classifier()` up to 5 minutes.
