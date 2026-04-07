# main.py

from src.config import *
from src.data_loader import load_data
from src.preprocessing import preparer_donnees
from src.features_engineering import construire_features
from src.multiple_regression import selection_backward, regression_multiple
from src.comparison_model import arbre_decision_regression
from src.evaluation import comparer_modeles, afficher_comparaison_graphique
from src.interface import lancer_interface


def executer_pipeline():
    """
    Exécute l'ensemble du pipeline data science :
    chargement → prétraitement → features → modèles → évaluation
    """

    print("\n=== LANCEMENT DU PIPELINE ARCHE ===")

    # 1. Chargement
    df_logs, df_notes = load_data()
    if df_logs is None or df_notes is None:
        print("Erreur lors du chargement des données.")
        return

    # 2. Prétraitement
    df_logs, df_notes = preparer_donnees(df_logs, df_notes)
    if df_logs is None or df_notes is None:
        print("Erreur lors du prétraitement.")
        return

    # 3. Feature engineering
    df_final = construire_features(df_logs, df_notes)
    print("Features construites :", df_final.shape)

    # 4. Régression multiple
    variables_retenues, _ = selection_backward(df_final)
    _, y_test_reg, y_pred_reg, _, _ = regression_multiple(df_final, variables_retenues)

    # 5. Arbre de décision
    _, _, y_test_tree, y_pred_tree, _, _ = arbre_decision_regression(df_final)

    # 6. Évaluation
    resultats = comparer_modeles(
        y_test_reg=y_test_reg,
        y_pred_reg=y_pred_reg,
        y_test_tree=y_test_tree,
        y_pred_tree=y_pred_tree,
    )

    afficher_comparaison_graphique(resultats)

    print("\n=== PIPELINE TERMINÉ ===")


def main():
    """
    Point d'entrée principal du projet
    """

    print("\n=== PROJET ARCHE - PRÉDICTION DE NOTE ===")

    # 1. Exécuter pipeline data science
    executer_pipeline()

    # 2. Lancer interface utilisateur
    print("\nLancement de l'interface utilisateur...")
    lancer_interface()


if __name__ == "__main__":
    main()