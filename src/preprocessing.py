'''
Description preprocessing.py
Projet : Prédiction de la note à partir des traces ARCHE

Module de prétraitement :
- nettoyage du fichier notes
- nettoyage du fichier logs
- mise en cohérence entre les deux sources
- audit simple de qualité des données

@author: Khady Diagne
@version: 1.0
@date: Avril 2026
'''

import pandas as pd
import unicodedata

from config import (
    LOGS_TIME_COL,
    LOGS_USER_COL,
    LOGS_CONTEXT_COL,
    LOGS_COMPONENT_COL,
    LOGS_EVENT_COL,
    NOTES_USER_COL,
    NOTES_TARGET_COL,
)


def _normaliser_texte(serie: pd.Series) -> pd.Series:
    '''
    Normalisation d'une série de texte
    '''
    return (
        serie.astype(str)
        .str.strip()
        .str.lower()
        .apply(
            lambda x: unicodedata.normalize("NFKD", x)
            .encode("ascii", "ignore")
            .decode("utf-8")
        )
    )


def _categoriser_evenement(evenement: str) -> str:
    '''
    Regroupement des événements en grandes catégories
    '''
    if "test" in evenement:
        return "test"
    elif "cours" in evenement or "module" in evenement:
        return "consultation"
    elif "discussion" in evenement or "contenu poste" in evenement:
        return "interaction"
    elif "visite guidee" in evenement:
        return "visite_guidee"
    elif "profil utilisateur" in evenement or "rapport" in evenement:
        return "navigation"
    else:
        return "autre"


def preparer_donnees(df_logs: pd.DataFrame, df_notes: pd.DataFrame):
    '''
    Prétraitement global des données
    :param df_logs: dataframe des logs
    :param df_notes: dataframe des notes
    :return: df_logs, df_notes nettoyés
    '''
    print("\nLANCEMENT DU PRÉTRAITEMENT DES DONNÉES...")

    if df_logs is None or df_notes is None:
        print("Erreur : les données ne sont pas disponibles.")
        return None, None

    try:
        df_notes = _traiter_notes(df_notes)
        df_logs = _traiter_logs(df_logs)
        df_logs = _filtrer_etudiants(df_logs, df_notes)

        print("\nPRÉTRAITEMENT DES DONNÉES TERMINÉ")
        return df_logs, df_notes

    except Exception as e:
        print(f"Erreur pendant le prétraitement : {e}")
        return None, None


def _traiter_notes(df_notes: pd.DataFrame) -> pd.DataFrame:
    '''
    Traitement du fichier notes
    '''
    print("\n[1] Analyse du fichier notes")

    data = df_notes.copy()

    # Conversion du pseudo en numérique
    data[NOTES_USER_COL] = pd.to_numeric(data[NOTES_USER_COL], errors="coerce")
    nb_pseudo_invalides = data[NOTES_USER_COL].isna().sum()
    print(f"Pseudos invalides dans notes : {nb_pseudo_invalides}")

    data = data.dropna(subset=[NOTES_USER_COL])
    data[NOTES_USER_COL] = data[NOTES_USER_COL].astype(int)

    # Suppression des doublons exacts
    nb_avant = len(data)
    data = data.drop_duplicates()
    nb_doublons = nb_avant - len(data)
    print(f"Lignes supprimées (doublons exacts) : {nb_doublons}")

    # Conversion de la note en numérique
    data[NOTES_TARGET_COL] = pd.to_numeric(data[NOTES_TARGET_COL], errors="coerce")
    nb_notes_invalides = data[NOTES_TARGET_COL].isna().sum()
    print(f"Notes invalides détectées : {nb_notes_invalides}")

    data = data.dropna(subset=[NOTES_TARGET_COL])

    print(f"Nombre final d'étudiants dans notes : {data[NOTES_USER_COL].nunique()}")

    return data


def _traiter_logs(df_logs: pd.DataFrame) -> pd.DataFrame:
    '''
    Traitement du fichier logs
    '''
    print("\n[2] Analyse du fichier logs")

    data = df_logs.copy()

    # Conversion du pseudo en numérique
    data[LOGS_USER_COL] = pd.to_numeric(data[LOGS_USER_COL], errors="coerce")
    nb_pseudo_invalides = data[LOGS_USER_COL].isna().sum()
    print(f"Pseudos invalides dans logs : {nb_pseudo_invalides}")

    data = data.dropna(subset=[LOGS_USER_COL])
    data[LOGS_USER_COL] = data[LOGS_USER_COL].astype(int)

    # Conversion de la date
    data[LOGS_TIME_COL] = pd.to_datetime(data[LOGS_TIME_COL], errors="coerce")
    nb_dates_invalides = data[LOGS_TIME_COL].isna().sum()
    print(f"Dates non exploitables : {nb_dates_invalides}")

    data = data.dropna(subset=[LOGS_TIME_COL])

    # Normalisation des colonnes texte
    for col in [LOGS_CONTEXT_COL, LOGS_COMPONENT_COL, LOGS_EVENT_COL]:
        data[col] = data[col].fillna("")
        data[col] = _normaliser_texte(data[col])

    # Suppression des lignes sans information exploitable
    masque_vides = (
        (data[LOGS_CONTEXT_COL] == "")
        & (data[LOGS_COMPONENT_COL] == "")
        & (data[LOGS_EVENT_COL] == "")
    )

    nb_vides = masque_vides.sum()
    print(f"Lignes sans information de connexion exploitable : {nb_vides}")

    data = data[~masque_vides]

    # Suppression des doublons exacts
    nb_avant = len(data)
    data = data.drop_duplicates()
    nb_doublons = nb_avant - len(data)
    print(f"Doublons supprimés : {nb_doublons}")

    # Construction d'une catégorie d'événement plus synthétique
    data["categorie_evenement"] = data[LOGS_EVENT_COL].apply(_categoriser_evenement)

    print("\nValeurs uniques principales après normalisation :")
    print("Composants :", sorted(data[LOGS_COMPONENT_COL].dropna().unique())[:10])
    print("Événements :", sorted(data[LOGS_EVENT_COL].dropna().unique())[:10])
    print("Catégories d'événements :", sorted(data["categorie_evenement"].dropna().unique()))

    print(f"Nombre final de lignes dans logs : {len(data)}")
    print(f"Nombre final d'étudiants distincts dans logs : {data[LOGS_USER_COL].nunique()}")

    return data


def _filtrer_etudiants(df_logs: pd.DataFrame, df_notes: pd.DataFrame) -> pd.DataFrame:
    '''
    Mise en cohérence entre logs et notes
    '''
    print("\n[3] Cohérence entre logs et notes")

    etudiants_notes = set(df_notes[NOTES_USER_COL])
    etudiants_logs = set(df_logs[LOGS_USER_COL])

    nb_notes = len(etudiants_notes)
    nb_logs_avant = len(etudiants_logs)

    print(f"Étudiants dans notes : {nb_notes}")
    print(f"Étudiants dans logs (avant filtre) : {nb_logs_avant}")

    # Conserver uniquement les étudiants présents dans le fichier notes
    # Cela permet de construire un jeu supervisé cohérent
    data = df_logs[df_logs[LOGS_USER_COL].isin(etudiants_notes)]

    nb_logs_apres = data[LOGS_USER_COL].nunique()
    nb_supprimes = nb_logs_avant - nb_logs_apres

    print(f"Étudiants conservés dans logs : {nb_logs_apres}")
    print(f"Étudiants supprimés car absents du fichier notes : {nb_supprimes}")

    # Étudiants présents dans notes mais sans activité dans logs
    etudiants_logs_apres = set(data[LOGS_USER_COL])
    nb_sans_activite = len(etudiants_notes - etudiants_logs_apres)

    print(f"Étudiants présents dans notes mais sans activité ARCHE : {nb_sans_activite}")

    return data


def audit_qualite_donnees(df_logs: pd.DataFrame, df_notes: pd.DataFrame) -> None:
    '''
    Audit simple de qualité des données
    '''
    print("\nAUDIT QUALITÉ DES DONNÉES")

    print("\nValeurs manquantes dans logs :")
    print(df_logs.isna().sum())

    print("\nValeurs manquantes dans notes :")
    print(df_notes.isna().sum())

    print("\nNotes hors intervalle [0, 20] :")
    print(((df_notes[NOTES_TARGET_COL] < 0) | (df_notes[NOTES_TARGET_COL] > 20)).sum())

    print("\nPseudos <= 0 dans notes :")
    print((df_notes[NOTES_USER_COL] <= 0).sum())

    print("\nPseudos <= 0 dans logs :")
    print((df_logs[LOGS_USER_COL] <= 0).sum())

    print("\nPlage des dates dans logs :")
    print("Min :", df_logs[LOGS_TIME_COL].min())
    print("Max :", df_logs[LOGS_TIME_COL].max())


if __name__ == "__main__":
    print("Test du prétraitement des données...")

    from data_loader import load_data

    df_logs, df_notes = load_data()

    if df_logs is not None and df_notes is not None:
        df_logs, df_notes = preparer_donnees(df_logs, df_notes)

        if df_logs is not None and df_notes is not None:
            print("\nPrétraitement réussi. Aperçu des données :")
            print(df_logs.head())
            print(df_notes.head())
            audit_qualite_donnees(df_logs, df_notes)
        else:
            print("Erreur lors du prétraitement.")
    else:
        print("Erreur lors du chargement des données.")