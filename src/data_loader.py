# data_loader.py

"""
Chargement sécurisé des données fournies

Ce module :
 - vérifie l'existence des fichiers,
 - charge les fichiers CSV,
 - contrôle les colonnes attendues,
 - gère les principales erreurs de lecture.
"""

import os
import pandas as pd

from config import (
    LOGS_PATH,
    NOTES_PATH,
    EXPECTED_LOGS_COLUMNS,
    EXPECTED_NOTES_COLUMNS,
)


def file_exists(file_path: str) -> bool:
    """
    Vérifie qu'un fichier existe.

    :param file_path: chemin du fichier
    :return: True si le fichier existe, False sinon
    """
    return os.path.exists(file_path)


def check_columns(df: pd.DataFrame, expected_columns: list[str], file_name: str) -> bool:
    """
    Vérifie que toutes les colonnes attendues sont présentes.

    :param df: DataFrame à contrôler
    :param expected_columns: liste des colonnes attendues
    :param file_name: nom du fichier pour affichage des erreurs
    :return: True si les colonnes sont présentes, False sinon
    """
    missing_columns = [col for col in expected_columns if col not in df.columns]

    if missing_columns:
        print(f"ERREUR : colonnes manquantes dans {file_name} : {missing_columns}")
        return False

    return True


def load_logs() -> pd.DataFrame | None:
    """
    Charge le fichier des logs.

    :return: DataFrame des logs ou None si erreur
    """
    if not file_exists(LOGS_PATH):
        print(f"ERREUR : fichier introuvable -> {LOGS_PATH}")
        return None

    try:
        df_logs = pd.read_csv(LOGS_PATH)

        if df_logs.empty:
            print("ERREUR : le fichier des logs est vide.")
            return None

        if not check_columns(df_logs, EXPECTED_LOGS_COLUMNS, "logs.csv"):
            return None

        print(f"Chargement réussi : logs.csv contient {len(df_logs)} lignes.")
        return df_logs

    except FileNotFoundError:
        print("ERREUR : le fichier logs.csv est introuvable.")
        return None

    except pd.errors.EmptyDataError:
        print("ERREUR : le fichier logs.csv est vide ou illisible.")
        return None

    except pd.errors.ParserError:
        print("ERREUR : problème de lecture du fichier logs.csv.")
        return None

    except PermissionError:
        print("ERREUR : le fichier logs.csv est ouvert dans un autre programme.")
        return None

    except Exception as unknown_error:
        print(f"ERREUR imprévue lors du chargement des logs : {unknown_error}")
        return None


def load_notes() -> pd.DataFrame | None:
    """
    Charge le fichier des notes.

    :return: DataFrame des notes ou None si erreur
    """
    if not file_exists(NOTES_PATH):
        print(f"ERREUR : fichier introuvable -> {NOTES_PATH}")
        return None

    try:
        df_notes = pd.read_csv(NOTES_PATH)

        if df_notes.empty:
            print("ERREUR : le fichier des notes est vide.")
            return None

        if not check_columns(df_notes, EXPECTED_NOTES_COLUMNS, "notes.csv"):
            return None

        print(f"Chargement réussi : notes.csv contient {len(df_notes)} lignes.")
        return df_notes

    except FileNotFoundError:
        print("ERREUR : le fichier notes.csv est introuvable.")
        return None

    except pd.errors.EmptyDataError:
        print("ERREUR : le fichier notes.csv est vide ou illisible.")
        return None

    except pd.errors.ParserError:
        print("ERREUR : problème de lecture du fichier notes.csv.")
        return None

    except PermissionError:
        print("ERREUR : le fichier notes.csv est ouvert dans un autre programme.")
        return None

    except Exception as unknown_error:
        print(f"ERREUR imprévue lors du chargement des notes : {unknown_error}")
        return None


def load_data() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """
    Charge les deux fichiers nécessaires au projet.

    :return: tuple (df_logs, df_notes)
    """
    print("CHARGEMENT DES DONNÉES...")

    df_logs = load_logs()
    df_notes = load_notes()

    if df_logs is None or df_notes is None:
        print("Le chargement des données a échoué.")
        return None, None

    print("Les deux fichiers ont été chargés avec succès.")
    return df_logs, df_notes


if __name__ == "__main__":
    logs, notes = load_data()

    if logs is not None and notes is not None:
        print("\nAperçu des logs :")
        print(logs.head())

        print("\nAperçu des notes :")
        print(notes.head())

        print("\nAffichage des types de données :")
        print(logs.dtypes)
        print(notes.dtypes)

        print("\nRésumé rapide des colonnes clés :")
        print(f"Nombre de contextes uniques : {logs['contexte'].nunique()}")
        print(f"Nombre de composants uniques : {logs['composant'].nunique()}")
        print(f"Nombre d'événements uniques : {logs['evenement'].nunique()}")

        print("\nExemples de valeurs :")
        print(f"Contextes : {logs['contexte'].dropna().unique()[:10]}")
        print(f"Composants : {logs['composant'].dropna().unique()[:10]}")
        print(f"Événements : {logs['evenement'].dropna().unique()[:10]}")
    else:
        print("Test interrompu : données non disponibles.")