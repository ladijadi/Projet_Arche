# -*- coding: utf-8 -*-
'''
Description exploration.py
Projet : Prédiction de la note à partir des traces ARCHE

Module d'exploration :
- exploration des données brutes
- exploration des variables construites
- visualisations descriptives

@author: Khady Diagne
@version: 1.1
@date: Avril 2026
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from config import (
    LOGS_USER_COL,
    LOGS_CONTEXT_COL,
    LOGS_EVENT_COL,
    LOGS_COMPONENT_COL,
    NOTES_USER_COL,
    NOTES_TARGET_COL,
    TOP_N_EVENTS,
)


def explorer_logs(df_logs: pd.DataFrame) -> pd.Series:
    '''
    Affiche un résumé descriptif du fichier logs.

    :param df_logs: dataframe des logs prétraités
    :return: série du nombre d'actions par étudiant
    '''
    print("\nEXPLORATION DES LOGS")

    nb_actions_total = len(df_logs)
    nb_etudiants_actifs = df_logs[LOGS_USER_COL].nunique()
    nb_contextes = df_logs[LOGS_CONTEXT_COL].nunique()

    print(f"Nombre total d'actions enregistrées : {nb_actions_total}")
    print(f"Nombre d'étudiants actifs dans les logs : {nb_etudiants_actifs}")
    print(f"Nombre de contextes distincts consultés : {nb_contextes}")

    print(f"\nTop {TOP_N_EVENTS} des événements les plus fréquents :")
    print(df_logs[LOGS_EVENT_COL].value_counts().head(TOP_N_EVENTS).to_string())

    actions_par_etudiant = df_logs.groupby(LOGS_USER_COL).size()

    print("\nRésumé du nombre d'actions par étudiant :")
    print(actions_par_etudiant.describe())

    return actions_par_etudiant


def explorer_notes(df_notes: pd.DataFrame) -> None:
    '''
    Affiche un résumé descriptif du fichier notes.

    :param df_notes: dataframe des notes prétraitées
    '''
    print("\nEXPLORATION DES NOTES")

    nb_etudiants = df_notes[NOTES_USER_COL].nunique()
    note_min = df_notes[NOTES_TARGET_COL].min()
    note_moyenne = df_notes[NOTES_TARGET_COL].mean()
    note_max = df_notes[NOTES_TARGET_COL].max()

    print(f"Nombre d'étudiants dans le fichier notes : {nb_etudiants}")
    print(f"Note minimale : {note_min}")
    print(f"Note moyenne : {note_moyenne:.2f}")
    print(f"Note maximale : {note_max}")

    print("\nRésumé statistique des notes :")
    print(df_notes[NOTES_TARGET_COL].describe())


def identifier_etudiants_sans_activite(
    df_logs: pd.DataFrame,
    df_notes: pd.DataFrame
) -> pd.DataFrame:
    '''
    Identifie les étudiants présents dans les notes mais absents des logs.
    '''
    print("\nÉTUDIANTS SANS ACTIVITÉ SUR ARCHE")

    pseudos_actifs = set(df_logs[LOGS_USER_COL].unique())
    pseudos_notes = set(df_notes[NOTES_USER_COL].unique())

    pseudos_absents = pseudos_notes - pseudos_actifs
    df_absents = df_notes[df_notes[NOTES_USER_COL].isin(pseudos_absents)].copy()

    print(f"Nombre d'étudiants sans activité : {len(df_absents)}")

    if not df_absents.empty:
        print("\nListe des étudiants sans activité :")
        print(df_absents[[NOTES_USER_COL, NOTES_TARGET_COL]].to_string(index=False))
    else:
        print("Tous les étudiants ayant une note possèdent au moins une activité.")

    return df_absents


def identifier_top_activites(df_logs: pd.DataFrame, df_notes: pd.DataFrame):
    '''
    Identifie les étudiants les plus actifs.
    '''
    actions_par_user = (
        df_logs.groupby(LOGS_USER_COL)
        .size()
        .reset_index(name="nb_actions")
        .sort_values(by="nb_actions", ascending=False)
    )

    print("\nTop 5 étudiants les plus actifs :")
    print(actions_par_user.head())

    top_user = actions_par_user.iloc[0][LOGS_USER_COL]
    nb_actions = actions_par_user.iloc[0]["nb_actions"]

    print(f"\nÉtudiant le plus actif : {top_user} avec {nb_actions} actions")

    note_user = df_notes[df_notes[NOTES_USER_COL] == top_user]

    if note_user.empty:
        print("Aucune note trouvée pour cet étudiant.")
    else:
        print(f"Note de cet étudiant : {note_user[NOTES_TARGET_COL].values[0]}")

    return actions_par_user, top_user


def analyser_relation_activite_note(
    df_logs: pd.DataFrame,
    df_notes: pd.DataFrame
) -> pd.DataFrame:
    '''
    Construit un tableau liant nombre d'actions et note,
    puis calcule la corrélation linéaire.
    '''
    print("\nRELATION ENTRE ACTIVITÉ ET NOTE")

    df_actions = df_logs.groupby(LOGS_USER_COL).size().reset_index(name="nb_actions")

    df_relation = df_actions.merge(
        df_notes,
        left_on=LOGS_USER_COL,
        right_on=NOTES_USER_COL,
        how="inner"
    )

    correlation = df_relation["nb_actions"].corr(df_relation[NOTES_TARGET_COL])

    if pd.isna(correlation):
        print("La corrélation n'a pas pu être calculée.")
    else:
        print(f"Corrélation entre nb_actions et note : {correlation:.3f}")

        if abs(correlation) < 0.2:
            print("Le lien linéaire entre activité brute et note semble faible.")
        else:
            print("Un signal intéressant apparaît entre activité brute et note.")

    return df_relation


def afficher_graphiques_bruts(
    df_logs: pd.DataFrame,
    df_notes: pd.DataFrame,
    df_relation: pd.DataFrame
) -> None:
    '''
    Affiche des graphiques descriptifs de base.
    '''
    actions_par_etudiant = df_logs.groupby(LOGS_USER_COL).size()

    plt.figure(figsize=(8, 5))
    plt.hist(actions_par_etudiant, bins=20)
    plt.title("Distribution du nombre d'actions par étudiant")
    plt.xlabel("Nombre d'actions")
    plt.ylabel("Effectif")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.hist(df_notes[NOTES_TARGET_COL], bins=15)
    plt.title("Distribution des notes")
    plt.xlabel("Note")
    plt.ylabel("Effectif")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.scatter(df_relation["nb_actions"], df_relation[NOTES_TARGET_COL])
    plt.title("Relation entre nombre d'actions et note")
    plt.xlabel("Nombre d'actions")
    plt.ylabel("Note")
    plt.tight_layout()
    plt.show()


def afficher_top_evenements(df_logs: pd.DataFrame) -> None:
    '''
    Affiche les événements les plus fréquents.
    '''
    top_events = df_logs[LOGS_EVENT_COL].value_counts().head(TOP_N_EVENTS)

    plt.figure(figsize=(10, 5))
    plt.bar(top_events.index, top_events.values)
    plt.title(f"Top {TOP_N_EVENTS} des événements les plus fréquents")
    plt.xlabel("Événement")
    plt.ylabel("Fréquence")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()


def afficher_categories_evenement(df_logs: pd.DataFrame) -> None:
    '''
    Affiche la répartition des catégories d'événements.
    '''
    print("\nRÉPARTITION DES CATÉGORIES D'ÉVÉNEMENTS")
    print(df_logs["categorie_evenement"].value_counts().to_string())

    top_cat = df_logs["categorie_evenement"].value_counts()

    plt.figure(figsize=(8, 5))
    plt.bar(top_cat.index, top_cat.values)
    plt.title("Répartition des catégories d'événements")
    plt.xlabel("Catégorie")
    plt.ylabel("Fréquence")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()


def afficher_composants(df_logs: pd.DataFrame) -> None:
    '''
    Affiche la répartition des composants.
    '''
    print("\nRÉPARTITION DES COMPOSANTS")
    print(df_logs[LOGS_COMPONENT_COL].value_counts().to_string())


def explorer_nb_contextes_par_etudiant(df_logs: pd.DataFrame) -> None:
    '''
    Explore la distribution du nombre de contextes par étudiant.
    '''
    print("\nDISTRIBUTION DU NOMBRE DE CONTEXTES PAR ÉTUDIANT")

    nb_contextes_par_user = df_logs.groupby(LOGS_USER_COL)[LOGS_CONTEXT_COL].nunique()
    print(nb_contextes_par_user.describe())

    plt.figure(figsize=(8, 5))
    plt.hist(nb_contextes_par_user, bins=20)
    plt.title("Distribution du nombre de contextes par étudiant")
    plt.xlabel("Nombre de contextes")
    plt.ylabel("Effectif")
    plt.tight_layout()
    plt.show()


def lancer_exploration(df_logs: pd.DataFrame, df_notes: pd.DataFrame) -> None:
    '''
    Lance l'exploration des données brutes.
    '''
    print("\nLANCEMENT DE L'EXPLORATION DES DONNÉES BRUTES...")

    explorer_logs(df_logs)
    explorer_notes(df_notes)
    identifier_etudiants_sans_activite(df_logs, df_notes)
    identifier_top_activites(df_logs, df_notes)
    afficher_top_evenements(df_logs)
    afficher_categories_evenement(df_logs)
    afficher_composants(df_logs)
    explorer_nb_contextes_par_etudiant(df_logs)

    df_relation = analyser_relation_activite_note(df_logs, df_notes)
    afficher_graphiques_bruts(df_logs, df_notes, df_relation)

    print("\nEXPLORATION DES DONNÉES BRUTES TERMINÉE")


def afficher_correlation_note(df_features: pd.DataFrame) -> pd.Series:
    '''
    Affiche la corrélation des features avec la note.
    '''
    print("\nCORRÉLATION DES FEATURES AVEC LA NOTE")

    corr_note = (
        df_features.corr(numeric_only=True)[NOTES_TARGET_COL]
        .sort_values(ascending=False)
    )

    print(corr_note)
    return corr_note


def afficher_matrice_correlation(df_features: pd.DataFrame) -> None:
    '''
    Affiche la matrice de corrélation des features.
    '''
    print("\nAFFICHAGE DE LA MATRICE DE CORRÉLATION...")

    variables = [
        NOTES_TARGET_COL,
        "nb_contextes",
        "nb_actions",
        "nb_jours_actifs",
        "temps_moyen_action",
        "temps_moyen_jour_actif",
        "ratio_test",
        "ratio_interaction",
        "ratio_consultation",
        "engagement_actif",
        "ratio_fichier"
    ]

    variables_presentes = [col for col in variables if col in df_features.columns]
    corr = df_features[variables_presentes].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(cax)

    ax.set_xticks(np.arange(len(variables_presentes)))
    ax.set_yticks(np.arange(len(variables_presentes)))
    ax.set_xticklabels(variables_presentes, rotation=45, ha="right")
    ax.set_yticklabels(variables_presentes)

    for i in range(len(variables_presentes)):
        for j in range(len(variables_presentes)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center")

    plt.title("Matrice de corrélation des features")
    plt.tight_layout()
    plt.show()

    print("\nMATRICE DE CORRÉLATION AFFICHÉE.")


def lancer_exploration_features(df_final: pd.DataFrame) -> None:
    '''
    Lance l'exploration des variables construites.
    '''
    print("\nLANCEMENT DE L'EXPLORATION DES FEATURES...")
    afficher_correlation_note(df_final)
    afficher_matrice_correlation(df_final)
    print("\nEXPLORATION DES FEATURES TERMINÉE")


if __name__ == "__main__":
    from data_loader import load_data
    from preprocessing import preparer_donnees
    from features_engineering import construire_features

    print("TEST DU MODULE EXPLORATION")

    df_logs, df_notes = load_data()

    if df_logs is None or df_notes is None:
        print("Erreur : chargement impossible.")
    else:
        df_logs, df_notes = preparer_donnees(df_logs, df_notes)

        if df_logs is None or df_notes is None:
            print("Erreur : prétraitement impossible.")
        else:
            lancer_exploration(df_logs, df_notes)

            df_final = construire_features(df_logs, df_notes)
            lancer_exploration_features(df_final)