# main.py

"""
Point d'entrée du projet ARCHE.

Ce script :
- charge les données
- applique le prétraitement
- affiche un résumé
"""

from data_loader import load_data
from preprocessing import preparer_donnees


def main():
    print("DÉMARRAGE DU PROGRAMME")

    # 1. Chargement des données
    df_logs, df_notes = load_data()

    if df_logs is None or df_notes is None:
        print("Arrêt du programme : chargement des données échoué.")
        return

    # 2. Prétraitement
    df_logs, df_notes = preparer_donnees(df_logs, df_notes)

    if df_logs is None or df_notes is None:
        print("Arrêt du programme : erreur lors du prétraitement.")
        return

    # 3. Résumé rapide
    print("\nRÉSUMÉ FINAL")
    print(f"Nombre de lignes logs : {len(df_logs)}")
    print(f"Nombre d'étudiants (logs) : {df_logs['pseudo'].nunique()}")
    print(f"Nombre d'étudiants (notes) : {df_notes['pseudo'].nunique()}")

    print("\nProgramme exécuté avec succès.")


if __name__ == "__main__":
    main()