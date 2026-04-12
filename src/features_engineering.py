# features_engineering.py
# -*- coding: utf-8 -*-

'''
Description features_engineering.py
Projet : Prédiction de la note à partir des traces ARCHE

Module de construction des variables explicatives :
- variables structurelles
- variable temporelle
- variables comportementales
- variable liée au composant fichier

@author: Khady Diagne
@version: 1.0
@date: Avril 2026
'''

import numpy as np
import pandas as pd

from config import (
    LOGS_USER_COL,
    LOGS_TIME_COL,
    LOGS_CONTEXT_COL,
    LOGS_COMPONENT_COL,
    NOTES_USER_COL,
)


def construire_features(df_logs: pd.DataFrame, df_notes: pd.DataFrame) -> pd.DataFrame:
    '''
    Construction du jeu final de variables explicatives
    :param df_logs: dataframe des logs prétraités
    :param df_notes: dataframe des notes nettoyées
    :return: dataframe final agrégé par étudiant
    '''
    data = df_logs.copy()

    print("\nCONSTRUCTION DES FEATURES FINALES...")

    # 1. Variables structurelles
    nb_contextes = _calculer_nb_contextes(data)
    nb_actions = _calculer_nb_actions(data)
    nb_jours = _calculer_nb_jours_actifs(data)

    # 2. Variable temporelle
    df_temps = _calculer_temps_moyen_action(data, nb_actions)

    # 3. Variables comportementales
    df_cat = _calculer_ratios_comportementaux(data, nb_actions)

    # 4. Variable composant
    df_ratio = _calculer_ratio_fichier(data, nb_actions)

    # 5. Variable temps moyen par jour actif

    df_temps_jour = _calculer_temps_moyen_jour_actif(data)

    # 6. Fusion finale avec les notes
    df_final = df_notes.copy()

    features = [
        nb_contextes,
        nb_actions,
        nb_jours,
        df_temps,
        df_temps_jour,
        df_cat,
        df_ratio,
    ]

    for feat in features:
        df_final = df_final.merge(feat, on=NOTES_USER_COL, how="left")

    df_final = df_final.fillna(0)

    print(f"Features finales construites : {len(df_final)} étudiants")

    return df_final


def _calculer_nb_contextes(df_logs: pd.DataFrame) -> pd.DataFrame:
    '''
    Nombre de contextes distincts par étudiant
    '''
    res = (
        df_logs.groupby(LOGS_USER_COL)[LOGS_CONTEXT_COL]
        .nunique()
        .reset_index(name="nb_contextes")
    )
    return res


def _calculer_nb_actions(df_logs: pd.DataFrame) -> pd.DataFrame:
    '''
    Nombre total d'actions par étudiant
    '''
    res = (
        df_logs.groupby(LOGS_USER_COL)
        .size()
        .reset_index(name="nb_actions")
    )
    return res


def _calculer_nb_jours_actifs(df_logs: pd.DataFrame) -> pd.DataFrame:
    '''
    Nombre de jours actifs par étudiant
    '''
    data = df_logs.copy()
    data["jour"] = data[LOGS_TIME_COL].dt.date

    res = (
        data.groupby(LOGS_USER_COL)["jour"]
        .nunique()
        .reset_index(name="nb_jours_actifs")
    )
    return res


def _calculer_temps_moyen_action(df_logs: pd.DataFrame, nb_actions: pd.DataFrame) -> pd.DataFrame:
    '''
    Temps moyen d'activité par action
    '''
    data = df_logs.copy()

    # Les logs sont triés pour calculer les écarts de temps entre actions successives
    data = data.sort_values(by=[LOGS_USER_COL, LOGS_TIME_COL])

    data["ecart"] = (
        data.groupby(LOGS_USER_COL)[LOGS_TIME_COL]
        .diff()
        .dt.total_seconds()
    )

    # On conserve uniquement les écarts courts, considérés comme une activité continue
    petits_ecarts = data[(data["ecart"] > 0) & (data["ecart"] < 300)]

    temps_total = (
        petits_ecarts.groupby(LOGS_USER_COL)["ecart"]
        .sum()
        .reset_index(name="temps_total_sec")
    )

    res = nb_actions.merge(temps_total, on=LOGS_USER_COL, how="left")

    res["temps_moyen_action"] = np.where(
        res["nb_actions"] > 0,
        res["temps_total_sec"] / res["nb_actions"],
        0
    )

    res = res[[LOGS_USER_COL, "temps_moyen_action"]]

    return res

def _calculer_temps_moyen_jour_actif(df_logs: pd.DataFrame) -> pd.DataFrame:
    '''
    Temps moyen d'activité par jour actif
    '''
    data = df_logs.copy()

    # Tri des logs par étudiant et par date
    data = data.sort_values(by=[LOGS_USER_COL, LOGS_TIME_COL])

    # Ajout du jour
    data["jour"] = data[LOGS_TIME_COL].dt.date

    # Calcul des écarts entre actions successives au sein d'un même jour
    data["ecart"] = (
        data.groupby([LOGS_USER_COL, "jour"])[LOGS_TIME_COL]
        .diff()
        .dt.total_seconds()
    )

    # Conservation des écarts courts assimilables à une activité continue
    petits_ecarts = data[(data["ecart"] > 0) & (data["ecart"] < 300)]

    # Temps total par étudiant et par jour
    temps_par_jour = (
        petits_ecarts.groupby([LOGS_USER_COL, "jour"])["ecart"]
        .sum()
        .reset_index(name="temps_jour_sec")
    )

    # Temps moyen d'activité par jour actif
    res = (
        temps_par_jour.groupby(LOGS_USER_COL)["temps_jour_sec"]
        .mean()
        .reset_index(name="temps_moyen_jour_actif")
    )

    return res


def _calculer_ratios_comportementaux(df_logs: pd.DataFrame, nb_actions: pd.DataFrame) -> pd.DataFrame:
    '''
    Calcul des ratios comportementaux à partir de categorie_evenement
    '''
    data = df_logs.copy()

    res = (
        data.groupby([LOGS_USER_COL, "categorie_evenement"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    # Certaines catégories peuvent être absentes selon les données
    for col in ["test", "interaction", "consultation"]:
        if col not in res.columns:
            res[col] = 0

    res = res.merge(nb_actions, on=LOGS_USER_COL, how="left")

    res["ratio_test"] = res["test"] / res["nb_actions"]
    res["ratio_interaction"] = res["interaction"] / res["nb_actions"]
    res["ratio_consultation"] = res["consultation"] / res["nb_actions"]

    # Indicateur synthétique d'engagement actif
    res["engagement_actif"] = (res["test"] + res["interaction"]) / res["nb_actions"]

    res = res[
        [
            LOGS_USER_COL,
            "ratio_test",
            "ratio_interaction",
            "ratio_consultation",
            "engagement_actif",
        ]
    ]

    return res


def _calculer_ratio_fichier(df_logs: pd.DataFrame, nb_actions: pd.DataFrame) -> pd.DataFrame:
    '''
    Calcul du ratio d'actions sur le composant fichier
    '''
    data = df_logs.copy()

    nb_fichier = (
        data[data[LOGS_COMPONENT_COL].str.lower() == "fichier"]
        .groupby(LOGS_USER_COL)
        .size()
        .reset_index(name="nb_fichier")
    )

    res = nb_actions.merge(nb_fichier, on=LOGS_USER_COL, how="left")
    res["nb_fichier"] = res["nb_fichier"].fillna(0)

    res["ratio_fichier"] = res["nb_fichier"] / res["nb_actions"]

    res = res[[LOGS_USER_COL, "ratio_fichier"]]

    return res


if __name__ == "__main__":
    print("Test du module features...")

    from data_loader import load_data
    from preprocessing import preparer_donnees

    df_logs, df_notes = load_data()

    if df_logs is not None and df_notes is not None:
        df_logs, df_notes = preparer_donnees(df_logs, df_notes)

        if df_logs is not None and df_notes is not None:
            df_final = construire_features(df_logs, df_notes)

            print("\nAperçu du jeu final :")
            print(df_final.head().to_string(index=False))

            print("\nColonnes du jeu final :")
            print(df_final.columns.tolist())
        else:
            print("Erreur : prétraitement impossible.")
    else:
        print("Erreur : chargement impossible.")