import os
import unicodedata
import re
import shutil
import sklearn.metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency


# Restore metadata dans son état d'origine
def copy_metadata(datadir):
    src_file = "metadata_copie.csv"
    dst_file = os.path.join(datadir, "metadata.csv")
    shutil.copyfile(src_file, dst_file)
    print(f"Fichier {src_file} copié vers {dst_file} avec succès.")


# Crée une colonne Labels avec "sain" ou "malade"
def initialize_labels(datadir, df=None):
    if df is None:
        df = pd.read_csv(f"{datadir}/metadata.csv")
    df["Labels"] = df["Finding Labels"].apply(lambda x: "Sain" if x == "No Finding" else "Malade")
    df.to_csv(f"{datadir}/metadata.csv", index=False)
    return df


# Crée une colonne Labels avec "jeune" ou "vieux"
def initialize_age_category(datadir, df=None):
    if df is None:
        df = pd.read_csv(f"{datadir}/metadata.csv")
    df["Age Category"] = df["Patient Age"].apply(lambda x: "vieux" if x > 50 else "jeune")
    df.to_csv(f"{datadir}/metadata.csv", index=False)
    return df


# Crée une colonne Labels avec "F_jeune", "F_vieux", "H_jeune", "H_vieux"
def initialize_combined_age_gender(datadir, df=None):
    if df is None:
        df = pd.read_csv(f"{datadir}/metadata.csv")
    df["Age+Gender"] = df["Patient Gender"] + "_" + df["Age Category"]
    df.to_csv(f"{datadir}/metadata.csv", index=False)
    return df


# Réinitialise les poids à 1
def initialize_weights(datadir, df=None):
    if df is None:
        df = pd.read_csv(f"{datadir}/metadata.csv")
    df["WEIGHTS"] = 1
    df.to_csv(f"{datadir}/metadata.csv", index=False)
    return df


# Vérifie la plage des valeurs dans le csv des metadata
def check_data(datadir, df=None):
    if df is None:
        df = pd.read_csv(f"{datadir}/metadata.csv")
    
    nan_gender = df["Patient Gender"].isna().sum()
    nan_age = df["Patient Age"].isna().sum()
    nan_position = df["View Position"].isna().sum()
    nan_followup = df["Follow-up #"].isna().sum()
    nan_id = df["Patient ID"].isna().sum()
    if nan_gender > 0:
        print(f"{nan_gender} valeur(s) manquante(s) dans 'Patient Gender'")
    else:
        print("Pas de valeur manquante dans 'Patient Gender'")
    if nan_age > 0:
        print(f"{nan_age} valeur(s) manquante(s) dans 'Patient Age'")
    else:
        print("Pas de valeur manquante dans 'Patient Age'")
    if nan_position > 0:
        print(f"{nan_position} valeur(s) manquante(s) dans 'Patient Gender'")
    else:
        print("Pas de valeur manquante dans 'Patient Gender'")
    if nan_followup > 0:
        print(f"{nan_followup} valeur(s) manquante(s) dans 'Follow-up #'")
    else:
        print("Pas de valeur manquante dans 'Follow-up #'")
    if nan_id > 0:
        print(f"{nan_id} valeur(s) manquante(s) dans 'Patient ID'")
    else:
        print("Pas de valeur manquante dans 'Patient ID'")

    # Vérification des bornes d'âge
    age_min = df["Patient Age"].min()
    age_max = df["Patient Age"].max()
    print(f"Âge minimum : {age_min}")
    print(f"Âge maximum : {age_max}")
        
    if age_min < 0 or age_max > 120:
        print("Valeurs d'âge potentiellement incohérentes détectées.")
    else:
        print("Les âges sont dans une plage raisonnable (0–120).")


# Fait la matrice de confusion
def confusion_matrix(df=None):
    if df is None:
        df = pd.read_csv(f"expe_log/preds.csv")
    
    y_true = df["labels"]
    y_pred = df["preds"]
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_true, y_pred).ravel()
    cm_reordered = np.array([[tp, fn], [fp, tn]])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm_reordered, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=['Prédit Malade', 'Prédit Sain'], yticklabels=['Malade', 'Sain'])
    plt.title("Matrice de confusion")
    plt.ylabel("Vérité terrain (label)")
    plt.xlabel("Prédiction du modèle")
    plt.tight_layout()
    plt.show()

    return tp, fn, fp, tn


# Donne la métrique "Equalized Odds"
def equalized_odds(col, df=None):
    """
    Affiche la différence absolue des Vrai Positif Rate et Faux Positif Rate entre les groupes
    """
    if df is None:
        df = pd.read_csv(f"expe_log/preds.csv")
    metrics = {}

    # Pour chaque genre, on calcule TPR et FPR
    for group in df[col].unique():
        group_df = df[df[col] == group]
        y_true = group_df["labels"]
        y_pred = group_df["preds"]
        
        # vrais positifs / total des positifs réels
        TP = len(group_df[(y_true == 'malade') & (y_pred == 'malade')])
        FN = len(group_df[(y_true == 'malade') & (y_pred == 'sain')])
        TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
        
        # faux positifs / total des négatifs réels
        FP = len(group_df[(y_true == 'sain') & (y_pred == 'malade')])
        TN = len(group_df[(y_true == 'sain') & (y_pred == 'sain')])
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
        
        metrics[group] = {"TPR": TPR, "FPR": FPR}
        print(f"Pour le groupe {group} : TPR = {TPR:.3f}, FPR = {FPR:.3f}")

    groups_list = list(metrics.keys())
    # Écart entre le maximum et le minimum
    all_TPR = [m["TPR"] for m in metrics.values()]
    all_FPR = [m["FPR"] for m in metrics.values()]
    diff_TPR = max(all_TPR) - min(all_TPR)
    diff_FPR = max(all_FPR) - min(all_FPR)
    print(f"Différence globale de TPR entre groupes: {diff_TPR:.3f}")
    print(f"Différence globale de FPR entre groupes: {diff_FPR:.3f}")

    # Groupe défavorisé en TPR : celui avec le TPR minimum
    disadvantaged_TPR = min(metrics.items(), key=lambda item: item[1]["TPR"])[0]
    # Groupe défavorisé en FPR : celui avec le FPR maximum
    disadvantaged_FPR = max(metrics.items(), key=lambda item: item[1]["FPR"])[0]
    print(f"Groupe défavorisé (TPR) : {disadvantaged_TPR}")
    print(f"Groupe défavorisé (FPR) : {disadvantaged_FPR}")
    
    return disadvantaged_TPR, disadvantaged_FPR


# Renvoie le score
def score(df):
    print(sklearn.metrics.balanced_accuracy_score(df.labels, df["preds"]))
    print(sklearn.metrics.accuracy_score(df.labels, df["preds"]))


# Fais une analyse bivariée
def bivariate_analysis(df, col, labels="Labels"):
    ct = pd.crosstab(df[col], df[labels])
    print("Effectif de chaque groupe par label:")
    print(ct)
    
    ct_percent = ct.div(ct.sum(axis=1), axis=0)
    print("Effectif de chaque groupe par label en %:")
    print(ct_percent)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(ct_percent, annot=True, cmap="Blues", fmt=".2f")
    plt.title(f"Répartition de '{col}' par '{labels}'")
    plt.ylabel(col)
    plt.xlabel(labels)
    # plt.show()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    folder = os.path.join(base_dir, "plots")
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder,  f"bivariate_analysis_{col}")
    plt.savefig(filepath, bbox_inches='tight', dpi=300)
    print(f"Affichage sauvegardé sous: {filepath}")
    
    chi2, p_value, dof, _= chi2_contingency(ct)
    print(f"Test du Chi2: chi2 = {chi2:.2f}, p-value = {p_value:.4f}, degrés de liberté = {dof}")
    return ct, ct_percent


# Analyse des biais en fonction d'une colonne
def analyze_bias_by_col(datadir, col, df=None):
    if df is None:
        df = pd.read_csv(f"{datadir}/metadata.csv")
    counts = df[col].value_counts()  # Compte le total de valeurs
    disease_counts = df.groupby([col, "Labels"]).size().unstack()  # Compte la répartion des valeurs par label

    # Affichage des statistiques
    print(f"Répartition de {col} dans le dataset: {counts}")
    print(f"Répartition des labels en fonction de {col}: {disease_counts}")

    # Visualisation des distributions
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.barplot(x=counts.index, y=counts.values, palette="Blues")
    plt.title(f"Distribution de {col}")
    plt.xlabel(f"{col}")
    plt.ylabel("Nombre d'images")

    plt.subplot(1, 2, 2)
    disease_counts.plot(kind="bar", stacked=True, colormap="coolwarm", ax=plt.gca())
    plt.title(f"Répartition des maladies par {col}")
    plt.xlabel(f"{col}")
    plt.ylabel("Nombre d'images")

    plt.tight_layout()
    plt.show()


# Pondère selon la répartition homme/femme dans le dataset
def reweight_by_group(col, datadir, df=None):
    if df is None:
        df = pd.read_csv(f"{datadir}/metadata.csv")

    gender_counts = df[col].value_counts()
    total_samples = len(df)
    weights = gender_counts.apply(lambda count: total_samples / (2 * count))
    df["WEIGHTS"] = df[col].map(weights)

    df.to_csv(f"{datadir}/metadata.csv", index=False)
    return df


# Pondère selon la répartition d'un groupe et leur distribution sain/malade dans le dataset
def reweight_by_group_and_label(col, datadir, df=None):
    if df is None:
        df = pd.read_csv(f"{datadir}/metadata.csv")

    group_counts = df.groupby([col, "Labels"]).size()
    num_groups = group_counts.shape[0]
    total_samples = len(df)
    average = total_samples / num_groups  # Effectif moyen par sous-groupe

    weights = group_counts.apply(lambda count: average / count)
    def assign_weight(row):
        return weights.loc[(row[col], row["Labels"])]
    df["WEIGHTS"] = df.apply(assign_weight, axis=1)

    df.to_csv(f"{datadir}/metadata.csv", index=False)
    return df


# Pondère selon la règle de Kamiran Calders
def reweight_kamiran_calders(col, datadir, df=None):
    if df is None:
        df = pd.read_csv(f"{datadir}/metadata.csv")
        
    total_samples = len(df)
    group_counts = df.groupby([col, "Labels"]).size()
    gender_counts = df.groupby(col).size()
    label_counts = df.groupby("Labels").size()
    
    # Calcul des poids pour chaque sous-groupe avec la formule : w(a, y) = (n_a * n_y) / (n_{a,y} * total_samples)
    weights = {}
    for gender in gender_counts.index:
        for label in label_counts.index:
            count_group = group_counts.get((gender, label), 0)
            n_gender = gender_counts[gender]
            n_label = label_counts[label]
            weights[(gender, label)] = (n_gender * n_label) / (count_group * total_samples)
    

    df["WEIGHTS"] = df.apply(lambda row: weights.get((row[col], row["Labels"]), 0), axis=1)
    df.to_csv(f"{datadir}/metadata.csv", index=False)
    return df


# Applique le post-processing 'Reject option'
def reject_option(col, df, disadvantaged_tp, disadvantaged_fp, confidence_threshold=0.55):
    # Si la prédiction est 'sain' mais l'intervalle de confiance est faible, 
    # et appartient au groupe défavorisé dans les True Positive, on switch à 'malade'
    mask_malade = ((df[col] == disadvantaged_tp) & (df["confidence"] < confidence_threshold) & 
                   (df["preds"] == "sain"))
    df.loc[mask_malade, "preds"] = "malade"
    df.loc[mask_malade, "confidence"] = 1

    # Si la prédiction est 'malade' mais l'intervalle de confiance est faible, 
    # et appartient au groupe défavorisé dans les False Positive, on switch à 'sain'
    mask_sain = ((df[col] == disadvantaged_fp) & (df["confidence"] < confidence_threshold) & 
                   (df["preds"] == "malade"))
    df.loc[mask_sain, "preds"] = "sain"
    df.loc[mask_sain, "confidence"] = 1

    df.to_csv(f"expe_log/preds.csv", index=False)
    return df


# # Applique le post-processing 'equalized odds'
def equalized_odds_postprocessing(col, df, disadvantaged_tp, disadvantaged_fp, confidence_threshold=0.55):
    metrics = {}
    groups = df[col].unique()
    for group in groups:
        group_df = df[df[col] == group]
        y_true = group_df["labels"]
        y_pred = group_df["preds"]
        
        TP = len(group_df[(y_true == "malade") & (y_pred == "malade")])
        FN = len(group_df[(y_true == "malade") & (y_pred == "sain")])
        TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
        
        FP = len(group_df[(y_true == "sain") & (y_pred == "malade")])
        TN = len(group_df[(y_true == "sain") & (y_pred == "sain")])
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
        
        metrics[group] = {"TPR": TPR, "FPR": FPR}
    
    advantaged_tp = [g for g in groups if g != disadvantaged_tp][0]
    advantaged_fp = [g for g in groups if g != disadvantaged_fp][0]
    
    TPR_diff = metrics[advantaged_tp]["TPR"] - metrics[disadvantaged_tp]["TPR"]
    FPR_diff = metrics[disadvantaged_fp]["FPR"] - metrics[advantaged_fp]["FPR"]
    
    p_flip_negative = (TPR_diff / (metrics[advantaged_tp]["TPR"] + 1e-6)) if metrics[advantaged_tp]["TPR"] > 0 else 0
    p_flip_positive = (FPR_diff / (metrics[disadvantaged_fp]["FPR"] + 1e-6)) if metrics[disadvantaged_fp]["FPR"] > 0 else 0

    p_flip_negative = min(max(p_flip_negative, 0), 1)
    p_flip_positive = min(max(p_flip_positive, 0), 1)
    
    
    # Si la prédiction est 'sain' mais l'intervalle de confiance est faible, 
    # et appartient au groupe défavorisé dans les True Positive, on switch à 'malade'
    mask_tp = ((df[col] == disadvantaged_tp) & (df["confidence"] < confidence_threshold))
    for idx, row in df[mask_tp].iterrows():
        if row["preds"] == "sain":
            if np.random.rand() < p_flip_negative:
                df.at[idx, "preds"] = "malade"
                df.at[idx, "confidence"] = 1

    # Si la prédiction est 'malade' mais l'intervalle de confiance est faible, 
    # et appartient au groupe défavorisé dans les False Positive, on switch à 'sain
    mask_fp = ((df[col] == disadvantaged_fp) & (df["confidence"] < confidence_threshold))
    for idx, row in df[mask_fp].iterrows():
        if row["preds"] == "malade":
            if np.random.rand() < p_flip_positive:
                df.at[idx, "preds"] = "sain"
                df.at[idx, "confidence"] = 1
    df.to_csv(f"expe_log/preds.csv", index=False)
    return df

# Fonction utilitaire pour la sauvegarde des graphes
def strip_accents(text):
    return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')


# Fonction utilitaire pour la sauvegarde des graphes
def extract_filename_from_title(title):
    title = strip_accents(title.lower())
    
    # Détermine la stratégie de reweighting
    if "kamiran" in title:
        method = "avec_reweighting_kamiran_calders"
    elif "classes" in title :
        method = "avec_reweighting_des_classes"
    else:
        method = "sans_reweighting"

    # Extraire 'avant' ou 'apres'
    match_phase = re.search(r'\b(avant|apres)\b', title)
    phase = match_phase.group(1) if match_phase else 'inconnu'

    # Extraire l'attribut (age, genre, age+genre)
    if "age+genre" in title:
        attr = "age_genre"
    elif "age" in title:
        attr = "age"
    elif "genre" in title:
        attr = "genre"
    else:
        attr = "inconnu"

    return f"{method}_{phase}_{attr}.png"


# Affiche les résultats des vrais positifs d'un classifieur
def show_tpr(y, x, title):
    plt.figure(figsize=(8, 6))
    bars = plt.bar(x, y, color='green')
    plt.xlabel("Groupes")
    plt.ylabel("True Positive Rate")
    plt.title(f"Taux de vrais positifs par groupe pour le {title}")
    plt.ylim(0, 1)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.02, f"{height:.3f}", ha='center', va='bottom')
    base_dir = os.path.dirname(os.path.abspath(__file__))
    folder = os.path.join(base_dir, "plots")
    os.makedirs(folder, exist_ok=True)
    filename = extract_filename_from_title(title)
    filepath = os.path.join(folder, f"vrais_positifs_{filename}")
    plt.savefig(filepath, bbox_inches='tight', dpi=300)
    print(f"Affichage sauvegardé sous: {filepath}")
    # plt.show()


def show_fpr(y, x, title):
    plt.figure(figsize=(8, 6))
    bars = plt.bar(x, y, color='red')
    plt.xlabel("Groupes")
    plt.ylabel("False Positive Rate")
    plt.title(f"Taux de faux positifs par groupe pour le {title}")
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.02, f"{height:.3f}", ha='center', va='bottom')
    base_dir = os.path.dirname(os.path.abspath(__file__))
    folder = os.path.join(base_dir, "plots")
    os.makedirs(folder, exist_ok=True)
    filename = extract_filename_from_title(title)
    filepath = os.path.join(folder,  f"faux_positifs_{filename}")
    plt.savefig(filepath, bbox_inches='tight', dpi=300)
    print(f"Affichage sauvegardé sous: {filepath}")
    # plt.show()


# Préparation au traitement
# col = "Patient Gender"
# col = "Age Category"
col = "Age+Gender"
datadir = "selected_data"  # Le dataset avec lequel travailler
copy_metadata(datadir)
df = pd.read_csv(f"{datadir}/metadata.csv")  # Le CSV des métadatas avec lequel travailler
# dfp = pd.read_csv(f"expe_log/preds.csv")  # Le CSV des prédictions avec lequel travailler
# score(dfp)
# initialize_labels(datadir, df)
# initialize_age_category(datadir, df)
# initialize_combined_age_gender(datadir, df)
# check_data(datadir, df)

# Reweighting (pré-processing):
# initialize_weights(datadir, df)
# reweight_by_group(col, datadir, df)
# reweight_by_group_and_label(col, datadir, df)
# reweight_kamiran_calders(col, datadir, df)

# Post-processing:
# disadvantaged_tp, disadvantaged_fp = equalized_odds(col)
# dfp_new = reject_option(col, dfp.copy(), disadvantaged_tp, disadvantaged_fp, confidence_threshold=0.555)
# dfp_new = equalized_odds_postprocessing(col, dfp.copy(), disadvantaged_tp, disadvantaged_fp, confidence_threshold=0.508)

# Métriques de fairness:
# bivariate_analysis(df, col)
# analyze_bias_by_col(datadir, col)
# confusion_matrix()
# equalized_odds(col)
# score(dfp_new)
# df.to_csv(f"{datadir}/metadata.csv", index=False)  # Pour récupérer les metadatas de base
# dfp.to_csv(f"expe_log/preds.csv", index=False)  # Pour récupérer les prédictions de base

# Visionnage des résultats:
# Biais sur le genre (cX = classifier utilisé, g|a|ag = genre|âge|âge+genre, 
#                     t|f = True|False positive rate, post = après post-processing)
"""
gender = ["Femme", "Homme"]
c0tg, c0fg = [0.440, 0.418], [0.076, 0.093]
c1t, c1f = [0.721, 0.661], [0.266, 0.272]
c2t, c2f = [0.836, 0.886], [0.434, 0.456]
c0tgpost, c0fgpost = [0.440, 0.439], [0.076, 0.095]
c1tpost, c1fpost = [0.721, 0.661], [0.266, 0.272]
c2tpost, c2fpost = [0.848, 0.881], [0.458, 0.508]
show_tpr(c0tg, gender, "classifier sans reweighting avant post-processing\nÉtude du biais sur le genre")
show_fpr(c0fg, gender, "classifier sans reweighting avant post-processing\nÉtude du biais sur le genre")
show_tpr(c1t, gender, "classifier avec reweighting des classes avant post-processing\nÉtude du biais sur le genre")
show_fpr(c1f, gender, "classifier avec reweighting des classes avant post-processing\nÉtude du biais sur le genre")
show_tpr(c2t, gender, "classifier avec reweighting Kamiran-Calders avant post-processing\nÉtude du biais sur le genre")
show_fpr(c2f, gender, "classifier avec reweighting Kamiran-Calders avant post-processing\nÉtude du biais sur le genre")
show_tpr(c0tgpost, gender, "classifier sans reweighting après post-processing\nÉtude du biais sur le genre")
show_fpr(c0fgpost, gender, "classifier sans reweighting après post-processing\nÉtude du biais sur le genre")
show_tpr(c1tpost, gender, "classifier avec reweighting des classes après post-processing\nÉtude du biais sur le genre")
show_fpr(c1fpost, gender, "classifier avec reweighting des classes après post-processing\nÉtude du biais sur le genre")
show_tpr(c2tpost, gender, "classifier avec reweighting Kamiran-Calders après post-processing\nÉtude du biais sur le genre")
show_fpr(c2fpost, gender, "classifier avec reweighting Kamiran-Calders après post-processing\nÉtude du biais sur le genre")

# Biais sur l'âge
age = ["Jeune", "Vieux"]
c0ta, c0fa = [0.399, 0.453], [0.070, 0.109]
c3t, c3f = [0.515, 0.560], [0.155, 0.199]
c4t, c4f = [0.668, 0.804], [0.276, 0.371]
c0tapost, c0fapost = [0.433, 0.434], [0.085, 0.079]
c3tpost, c3fpost = [0.530, 0.534], [0.179, 0.182]
c4tpost, c4fpost = [0.735, 0.735], [0.348, 0.291]
show_tpr(c0ta, age, "classifier sans reweighting avant post-processing\nÉtude du biais sur l'âge")
show_fpr(c0fa, age, "classifier sans reweighting avant post-processing\nÉtude du biais sur l'âge")
show_tpr(c3t, age, "classifier avec reweighting des classes avant post-processing\nÉtude du biais sur l'âge")
show_fpr(c3f, age, "classifier avec reweighting des classes avant post-processing\nÉtude du biais sur l'âge")
show_tpr(c4t, age, "classifier avec reweighting Kamiran-Calders avant post-processing\nÉtude du biais sur l'âge")
show_fpr(c4f, age, "classifier avec reweighting Kamiran-Calders avant post-processing\nÉtude du biais sur l'âge")
show_tpr(c0tapost, age, "classifier sans reweighting après post-processing\nÉtude du biais sur l'âge")
show_fpr(c0fapost, age, "classifier sans reweighting après post-processing\nÉtude du biais sur l'âge")
show_tpr(c3tpost, age, "classifier avec reweighting des classes après post-processing\nÉtude du biais sur l'âge")
show_fpr(c3fpost, age, "classifier avec reweighting des classes après post-processing\nÉtude du biais sur l'âge")
show_tpr(c4tpost, age, "classifier avec reweighting Kamiran-Calders après post-processing\nÉtude du biais sur l'âge")
show_fpr(c4fpost, age, "classifier avec reweighting Kamiran-Calders après post-processing\nÉtude du biais sur l'âge")

# Biais sur l'âge+genre
age_gender = ["Jeune Femme", "Vieille Femme", "Jeune Homme", "Vieil Homme"]
c0tag, c0fag = [0.385, 0.486, 0.411, 0.424], [0.073, 0.083, 0.068, 0.130]
c5t, c5f = [0.493, 0.646, 0.606, 0.652], [0.162, 0.158, 0.228, 0.302]
c6t, c6f = [0.682, 0.846, 0.756, 0.869], [0.296, 0.406, 0.376, 0.550]
c0tagpost, c0fagpost = [0.405, 0.486, 0.411, 0.409], [0.085, 0.083, 0.068, 0.089]
c5tpost, c5fpost = [0.581, 0.646, 0.606, 0.576], [0.211, 0.158, 0.228, 0.225]
c6tpost, c6fpost = [0.757, 0.846, 0.756, 0.747], [0.421, 0.406, 0.376, 0.432]
show_tpr(c0tag, age_gender, "classifier sans reweighting avant post-processing\nÉtude du biais sur l'âge+genre")
show_fpr(c0fag, age_gender, "classifier sans reweighting avant post-processing\nÉtude du biais sur l'âge+genre")
show_tpr(c5t, age_gender, "classifier avec reweighting des classes avant post-processing\nÉtude du biais sur l'âge+genre")
show_fpr(c5f, age_gender, "classifier avec reweighting des classes avant post-processing\nÉtude du biais sur l'âge+genre")
show_tpr(c6t, age_gender, "classifier avec reweighting Kamiran-Calders avant post-processing\nÉtude du biais sur l'âge+genre")
show_fpr(c6f, age_gender, "classifier avec reweighting Kamiran-Calders avant post-processing\nÉtude du biais sur l'âge+genre")
show_tpr(c0tagpost, age_gender, "classifier sans reweighting après post-processing\nÉtude du biais sur l'âge+genre")
show_fpr(c0fagpost, age_gender, "classifier sans reweighting après post-processing\nÉtude du biais sur l'âge+genre")
show_tpr(c5tpost, age_gender, "classifier avec reweighting des classes après post-processing\nÉtude du biais sur l'âge+genre")
show_fpr(c5fpost, age_gender, "classifier avec reweighting des classes après post-processing\nÉtude du biais sur l'âge+genre")
show_tpr(c6tpost, age_gender, "classifier avec reweighting Kamiran-Calders après post-processing\nÉtude du biais sur l'âge+genre")
show_fpr(c6fpost, age_gender, "classifier avec reweighting Kamiran-Calders après post-processing\nÉtude du biais sur l'âge+genre")
"""
