# preprocessing.py

import pandas as pd
from src.config import LOGS_TIME_COL, LOGS_USER_COL, NOTES_USER_COL, NOTES_TARGET_COL


def preparer_donnees(df_logs, df_notes):
    """
    Prépare et nettoie les données issues des logs et des notes.
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


# Traitement du fichier notes
def _traiter_notes(df_notes):

    print("\n[1] Analyse du fichier notes")

    total_avant = len(df_notes)

    df_notes = df_notes.drop_duplicates()

    total_apres = len(df_notes)
    nb_supprimes = total_avant - total_apres

    print(f"Lignes supprimées (doublons) : {nb_supprimes}")

    # Conversion en numérique avec gestion des erreurs
    df_notes[NOTES_TARGET_COL] = pd.to_numeric(df_notes[NOTES_TARGET_COL], errors="coerce")

    valeurs_invalides = df_notes[NOTES_TARGET_COL].isna().sum()

    print(f"Notes invalides détectées : {valeurs_invalides}")

    df_notes = df_notes.dropna(subset=[NOTES_TARGET_COL])

    return df_notes

# Traitement du fichier logs
def _traiter_logs(df_logs):

    print("\n[2] Analyse du fichier logs")

    df_logs[LOGS_TIME_COL] = pd.to_datetime(df_logs[LOGS_TIME_COL], errors="coerce")

    erreurs = df_logs[LOGS_TIME_COL].isna().sum()

    print(f"Dates non exploitables : {erreurs}")

    df_logs = df_logs.dropna(subset=[LOGS_TIME_COL])

    # Doublons
    avant = len(df_logs)
    df_logs = df_logs.drop_duplicates()
    apres = len(df_logs)

    print(f"Doublons supprimés : {avant - apres}")

    return df_logs

# Filtrage des étudiants pour ne garder que ceux présents dans les notes
def _filtrer_etudiants(df_logs, df_notes):

    print("\n[3] Cohérence entre logs et notes")

    etudiants_notes = set(df_notes[NOTES_USER_COL])
    etudiants_logs = set(df_logs[LOGS_USER_COL])

    print(f"Étudiants dans notes : {len(etudiants_notes)}")
    print(f"Étudiants dans logs (avant filtre) : {len(etudiants_logs)}")

    df_logs = df_logs[df_logs[LOGS_USER_COL].isin(etudiants_notes)]

    etudiants_logs_apres = df_logs[LOGS_USER_COL].nunique()

    print(f"Étudiants conservés dans logs : {etudiants_logs_apres}")

    return df_logs

if __name__ == "__main__":
    from src.data_loader import load_data

    df_logs, df_notes = load_data()
    df_logs_clean, df_notes_clean = preparer_donnees(df_logs, df_notes)

    print(df_logs_clean.head())
    print(df_notes_clean.head())