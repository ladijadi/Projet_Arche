# evaluation.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def calculer_metriques(y_true, y_pred) -> dict:
    """
    Calcule les métriques d'évaluation d'un modèle de régression.

    :param y_true: valeurs réelles
    :param y_pred: valeurs prédites
    :return: dictionnaire contenant MSE, RMSE, MAE et R²
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        "MSE": round(mse, 3),
        "RMSE": round(rmse, 3),
        "MAE": round(mae, 3),
        "R²": round(r2, 3),
    }


def comparer_modeles(
    y_test_reg,
    y_pred_reg,
    y_test_tree,
    y_pred_tree,
    nom_reg: str = "Régression multiple",
    nom_tree: str = "Arbre de décision régressif",
) -> pd.DataFrame:
    """
    Compare les performances de deux modèles de régression.

    :param y_test_reg: vraies valeurs pour la régression multiple
    :param y_pred_reg: prédictions de la régression multiple
    :param y_test_tree: vraies valeurs pour l'arbre
    :param y_pred_tree: prédictions de l'arbre
    :param nom_reg: nom du modèle linéaire
    :param nom_tree: nom du modèle arbre
    :return: DataFrame récapitulatif
    """
    metriques_reg = calculer_metriques(y_test_reg, y_pred_reg)
    metriques_tree = calculer_metriques(y_test_tree, y_pred_tree)

    resultats = pd.DataFrame([
        {
            "Modèle": nom_reg,
            "MSE": metriques_reg["MSE"],
            "RMSE": metriques_reg["RMSE"],
            "MAE": metriques_reg["MAE"],
            "R²": metriques_reg["R²"],
        },
        {
            "Modèle": nom_tree,
            "MSE": metriques_tree["MSE"],
            "RMSE": metriques_tree["RMSE"],
            "MAE": metriques_tree["MAE"],
            "R²": metriques_tree["R²"],
        }
    ])

    print("\nCOMPARAISON DES MODÈLES")
    print(resultats.to_string(index=False))

    print("\nINTERPRÉTATION")

    if (
        metriques_tree["RMSE"] < metriques_reg["RMSE"]
        and metriques_tree["R²"] > metriques_reg["R²"]
    ):
        print(
            f"Le modèle '{nom_tree}' est le plus performant selon les métriques principales "
            f"(RMSE plus faible et R² plus élevé)."
        )
    elif (
        metriques_reg["RMSE"] < metriques_tree["RMSE"]
        and metriques_reg["R²"] > metriques_tree["R²"]
    ):
        print(
            f"Le modèle '{nom_reg}' est le plus performant selon les métriques principales "
            f"(RMSE plus faible et R² plus élevé)."
        )
    else:
        print(
            "Les résultats sont partagés entre les deux modèles. "
            "Il faut alors arbitrer entre performance prédictive et interprétabilité."
        )

    print(
        "Le MAE permet aussi de mesurer l'erreur moyenne en points de note, "
        "ce qui donne une lecture plus concrète de la précision des modèles."
    )

    print(
        "La régression multiple reste plus facile à interpréter, "
        "tandis que l'arbre peut mieux capter des relations non linéaires."
    )

    diff_r2 = metriques_tree["R²"] - metriques_reg["R²"]
    diff_rmse = metriques_reg["RMSE"] - metriques_tree["RMSE"]

    print(f"\nGain R² (arbre vs régression) : {diff_r2:.3f}")
    print(f"Gain RMSE (réduction d'erreur) : {diff_rmse:.3f}")
    
    return resultats


def afficher_comparaison_graphique(resultats: pd.DataFrame) -> None:
    """
    Affiche des graphiques comparatifs des métriques.

    :param resultats: DataFrame de comparaison
    """
    for metrique in ["RMSE", "MAE", "R²"]:
        plt.figure(figsize=(8, 5))
        plt.bar(resultats["Modèle"], resultats[metrique])
        plt.title(f"Comparaison des {metrique}")
        plt.ylabel(metrique)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    from data_loader import load_data
    from preprocessing import preparer_donnees
    from features_engineering import construire_features
    from multiple_regression import selection_backward, regression_multiple
    from comparison_model import arbre_decision_regression

    print("Exécution de l'évaluation comparative...")

    df_logs, df_notes = load_data()

    if df_logs is not None and df_notes is not None:
        df_logs, df_notes = preparer_donnees(df_logs, df_notes)

        if df_logs is not None and df_notes is not None:
            df_final = construire_features(df_logs, df_notes)

            # Régression multiple
            variables_retenues, _ = selection_backward(df_final)
            _, y_test_reg, y_pred_reg, _, _ = regression_multiple(df_final, variables_retenues)

            # Arbre de décision régressif
            _, _, y_test_tree, y_pred_tree, _, _ = arbre_decision_regression(df_final)

            # Comparaison finale
            resultats = comparer_modeles(
                y_test_reg=y_test_reg,
                y_pred_reg=y_pred_reg,
                y_test_tree=y_test_tree,
                y_pred_tree=y_pred_tree,
            )

            afficher_comparaison_graphique(resultats)

        else:
            print("Erreur : prétraitement impossible.")
    else:
        print("Erreur : chargement impossible.")