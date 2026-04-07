# preprocessing.py

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
    """
    Nettoie une série texte :
    - conversion en chaîne
    - suppression espaces inutiles
    - passage en minuscules
    - suppression des accents
    """
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
    """
    Regroupe les événements en grandes catégories métier
    pour simplifier l'analyse comportementale.
    """
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
    """
    Lance l'ensemble du prétraitement :
    - nettoyage des notes
    - nettoyage des logs
    - cohérence entre logs et notes

    :param df_logs: DataFrame brut des logs
    :param df_notes: DataFrame brut des notes
    :return: (df_logs_pret, df_notes_pret) ou (None, None) si erreur
    """
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
    """
    Nettoie le fichier notes :
    - pseudo numérique
    - suppression des lignes invalides
    - suppression des doublons
    - conversion de la note en numérique
    """
    print("\n[1] Analyse du fichier notes")

    df_notes = df_notes.copy()

    # Uniformiser la colonne pseudo
    df_notes[NOTES_USER_COL] = pd.to_numeric(df_notes[NOTES_USER_COL], errors="coerce")
    nb_pseudo_invalides = df_notes[NOTES_USER_COL].isna().sum()
    print(f"Pseudos invalides dans notes : {nb_pseudo_invalides}")

    df_notes = df_notes.dropna(subset=[NOTES_USER_COL])
    df_notes[NOTES_USER_COL] = df_notes[NOTES_USER_COL].astype(int)

    # Supprimer les doublons exacts
    total_avant = len(df_notes)
    df_notes = df_notes.drop_duplicates()
    nb_supprimes = total_avant - len(df_notes)
    print(f"Lignes supprimées (doublons exacts) : {nb_supprimes}")

    # Uniformiser la note
    df_notes[NOTES_TARGET_COL] = pd.to_numeric(df_notes[NOTES_TARGET_COL], errors="coerce")
    nb_notes_invalides = df_notes[NOTES_TARGET_COL].isna().sum()
    print(f"Notes invalides détectées : {nb_notes_invalides}")

    df_notes = df_notes.dropna(subset=[NOTES_TARGET_COL])

    print(f"Nombre final d'étudiants dans notes : {df_notes[NOTES_USER_COL].nunique()}")

    return df_notes


def _traiter_logs(df_logs: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie le fichier logs :
    - pseudo numérique
    - dates valides
    - normalisation texte
    - suppression des lignes vides
    - suppression des doublons
    - création d'une catégorie d'événement
    """
    print("\n[2] Analyse du fichier logs")

    df_logs = df_logs.copy()

    # Uniformiser pseudo
    df_logs[LOGS_USER_COL] = pd.to_numeric(df_logs[LOGS_USER_COL], errors="coerce")
    nb_pseudo_invalides = df_logs[LOGS_USER_COL].isna().sum()
    print(f"Pseudos invalides dans logs : {nb_pseudo_invalides}")

    df_logs = df_logs.dropna(subset=[LOGS_USER_COL])
    df_logs[LOGS_USER_COL] = df_logs[LOGS_USER_COL].astype(int)

    # Uniformiser dates
    df_logs[LOGS_TIME_COL] = pd.to_datetime(df_logs[LOGS_TIME_COL], errors="coerce")
    nb_dates_invalides = df_logs[LOGS_TIME_COL].isna().sum()
    print(f"Dates non exploitables : {nb_dates_invalides}")

    df_logs = df_logs.dropna(subset=[LOGS_TIME_COL])

    # Normaliser colonnes texte
    for col in [LOGS_CONTEXT_COL, LOGS_COMPONENT_COL, LOGS_EVENT_COL]:
        df_logs[col] = df_logs[col].fillna("")
        df_logs[col] = _normaliser_texte(df_logs[col])

    # Supprimer les lignes sans information exploitable
    masque_vides = (
        (df_logs[LOGS_CONTEXT_COL] == "")
        & (df_logs[LOGS_COMPONENT_COL] == "")
        & (df_logs[LOGS_EVENT_COL] == "")
    )

    nb_vides = masque_vides.sum()
    print(f"Lignes sans information de connexion exploitable : {nb_vides}")

    df_logs = df_logs[~masque_vides]

    # Supprimer les doublons exacts
    nb_avant = len(df_logs)
    df_logs = df_logs.drop_duplicates()
    nb_doublons = nb_avant - len(df_logs)
    print(f"Doublons supprimés : {nb_doublons}")

    # Catégorisation des événements
    df_logs["categorie_evenement"] = df_logs[LOGS_EVENT_COL].apply(_categoriser_evenement)

    print("\nValeurs uniques principales après normalisation :")
    print("Composants :", sorted(df_logs[LOGS_COMPONENT_COL].dropna().unique())[:10])
    print("Événements :", sorted(df_logs[LOGS_EVENT_COL].dropna().unique())[:10])
    print("Catégories d'événements :", sorted(df_logs["categorie_evenement"].dropna().unique()))

    print(f"Nombre final de lignes dans logs : {len(df_logs)}")
    print(f"Nombre final d'étudiants distincts dans logs : {df_logs[LOGS_USER_COL].nunique()}")

    return df_logs


def _filtrer_etudiants(df_logs: pd.DataFrame, df_notes: pd.DataFrame) -> pd.DataFrame:
    """
    Ne conserve dans les logs que les étudiants présents dans le fichier notes.
    Cela permet de construire un jeu supervisé cohérent.
    """
    print("\n[3] Cohérence entre logs et notes")

    etudiants_notes = set(df_notes[NOTES_USER_COL])
    etudiants_logs = set(df_logs[LOGS_USER_COL])

    nb_notes = len(etudiants_notes)
    nb_logs_avant = len(etudiants_logs)

    print(f"Étudiants dans notes : {nb_notes}")
    print(f"Étudiants dans logs (avant filtre) : {nb_logs_avant}")

    # Conserver uniquement les étudiants pour lesquels on possède une note
    df_logs = df_logs[df_logs[LOGS_USER_COL].isin(etudiants_notes)]

    nb_logs_apres = df_logs[LOGS_USER_COL].nunique()
    nb_supprimes = nb_logs_avant - nb_logs_apres

    print(f"Étudiants conservés dans logs : {nb_logs_apres}")
    print(f"Étudiants supprimés car absents du fichier notes : {nb_supprimes}")

    # Étudiants sans activité ARCHE mais présents dans notes
    etudiants_logs_apres = set(df_logs[LOGS_USER_COL])
    nb_sans_activite = len(etudiants_notes - etudiants_logs_apres)

    print(f"Étudiants présents dans notes mais sans activité ARCHE : {nb_sans_activite}")

    return df_logs


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
        else:
            print("Erreur lors du prétraitement.")
    else:
        print("Erreur lors du chargement des données.")