# evaluation.py
# -*- coding: utf-8 -*-
'''
Description evaluation.py
Projet : Prédiction de la note à partir des traces ARCHE

Module d'évaluation comparative :
- calcul des métriques
- comparaison des modèles
- affichage graphique des résultats
- analyse visuelle des prédictions
- analyse des résidus

@author: Khady Diagne
@version: 2.0
@date: Avril 2026
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def calculer_metriques(y_true, y_pred) -> dict:
    '''
    Calcul des métriques d'un modèle de régression.

    :param y_true: valeurs réelles
    :param y_pred: valeurs prédites
    :return: dictionnaire des métriques
    '''
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
    '''
    Compare deux modèles de régression.

    :param y_test_reg: vraies valeurs pour la régression
    :param y_pred_reg: prédictions de la régression
    :param y_test_tree: vraies valeurs pour l'arbre
    :param y_pred_tree: prédictions de l'arbre
    :param nom_reg: nom du modèle de régression
    :param nom_tree: nom du modèle comparatif
    :return: tableau comparatif des métriques
    '''
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

    print("\nINTERPRÉTATION DES RÉSULTATS")

    if (
        metriques_tree["RMSE"] < metriques_reg["RMSE"]
        and metriques_tree["R²"] > metriques_reg["R²"]
    ):
        print(
            f"Le modèle '{nom_tree}' semble plus performant "
            f"(RMSE plus faible et R² plus élevé)."
        )
    elif (
        metriques_reg["RMSE"] < metriques_tree["RMSE"]
        and metriques_reg["R²"] > metriques_tree["R²"]
    ):
        print(
            f"Le modèle '{nom_reg}' semble plus performant "
            f"(RMSE plus faible et R² plus élevé)."
        )
    else:
        print(
            "Les résultats sont partagés. "
            "Il faut alors arbitrer entre précision prédictive et interprétabilité."
        )

    print(
        "Le MAE complète la lecture en indiquant l'erreur moyenne "
        "en points de note."
    )

    print(
        "La régression multiple reste plus interprétable, "
        "tandis que l'arbre permet de représenter des seuils de décision."
    )

    return resultats


def afficher_comparaison_graphique(resultats: pd.DataFrame) -> None:
    '''
    Affiche une comparaison graphique des métriques principales.

    :param resultats: tableau des résultats
    '''
    metriques = ["RMSE", "MAE", "R²"]

    plt.figure(figsize=(10, 6))

    x = np.arange(len(resultats["Modèle"]))
    largeur = 0.22

    for i, metrique in enumerate(metriques):
        plt.bar(
            x + i * largeur,
            resultats[metrique],
            width=largeur,
            label=metrique
        )

    plt.xticks(x + largeur, resultats["Modèle"], rotation=20)
    plt.ylabel("Valeur de la métrique")
    plt.title("Comparaison des métriques de performance par modèle")
    plt.legend()
    plt.tight_layout()
    plt.show()


def afficher_reel_vs_predit(y_true, y_pred, titre: str) -> None:
    '''
    Affiche un nuage de points entre valeurs réelles et prédites.

    :param y_true: valeurs réelles
    :param y_pred: valeurs prédites
    :param titre: titre du graphique
    '''
    plt.figure(figsize=(7, 5))
    plt.scatter(y_true, y_pred, alpha=0.7)

    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))

    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        linestyle="--",
        label="Prédiction parfaite (y=x)"
    )

    plt.xlabel("Valeurs réelles")
    plt.ylabel("Valeurs prédites")
    plt.title(titre)
    plt.legend()
    plt.tight_layout()
    plt.show()


def afficher_residus(y_true, y_pred, titre: str) -> None:
    '''
    Affiche la distribution des résidus.

    :param y_true: valeurs réelles
    :param y_pred: valeurs prédites
    :param titre: titre du graphique
    '''
    residus = np.array(y_true) - np.array(y_pred)

    mu = residus.mean()
    sigma = residus.std()

    plt.figure(figsize=(7, 5))
    plt.hist(residus, bins=20, density=True, alpha=0.7, edgecolor="black")

    if sigma > 0:
        x = np.linspace(residus.min(), residus.max(), 200)
        y = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        plt.plot(x, y, linewidth=2, label=f"Loi normale approchée (μ={mu:.3f}, σ={sigma:.3f})")

    plt.axvline(0, linestyle="--", label="Résidu nul")
    plt.axvline(mu, linestyle="--", label=f"Résidu moyen ({mu:.3f})")

    plt.xlabel("Résidus (Réel - Prédit)")
    plt.ylabel("Densité de probabilité")
    plt.title(titre)
    plt.legend()
    plt.tight_layout()
    plt.show()


def afficher_diagnostics_modeles(
    y_test_reg,
    y_pred_reg,
    y_test_tree,
    y_pred_tree
) -> None:
    '''
    Affiche les graphiques de diagnostic des deux modèles.

    :param y_test_reg: valeurs réelles pour la régression
    :param y_pred_reg: prédictions pour la régression
    :param y_test_tree: valeurs réelles pour l'arbre
    :param y_pred_tree: prédictions pour l'arbre
    '''
    afficher_reel_vs_predit(
        y_test_reg,
        y_pred_reg,
        "Régression multiple - valeurs réelles vs prédites"
    )

    afficher_reel_vs_predit(
        y_test_tree,
        y_pred_tree,
        "Arbre de décision - valeurs réelles vs prédites"
    )

    afficher_residus(
        y_test_reg,
        y_pred_reg,
        "Distribution des résidus - régression multiple"
    )

    afficher_residus(
        y_test_tree,
        y_pred_tree,
        "Distribution des résidus - arbre de décision"
    )


if __name__ == "__main__":
    print("TEST DU MODULE D'ÉVALUATION COMPARATIVE...")

    from data_loader import load_data
    from preprocessing import preparer_donnees
    from features_engineering import construire_features
    from multiple_regression import selection_backward, regression_multiple
    from comparison_model import arbre_decision_regression

    df_logs, df_notes = load_data()

    if df_logs is not None and df_notes is not None:
        df_logs, df_notes = preparer_donnees(df_logs, df_notes)

        if df_logs is not None and df_notes is not None:
            df_final = construire_features(df_logs, df_notes)

            variables_retenues, _ = selection_backward(df_final)

            _, y_test_reg, y_pred_reg, _, _ = regression_multiple(
                df_final,
                variables_retenues
            )

            (
                _,
                _,
                y_test_tree,
                y_pred_tree,
                _,
                _,
                _,
                _
            ) = arbre_decision_regression(
                df_final,
                variables=variables_retenues
            )

            resultats = comparer_modeles(
                y_test_reg=y_test_reg,
                y_pred_reg=y_pred_reg,
                y_test_tree=y_test_tree,
                y_pred_tree=y_pred_tree,
            )

            afficher_comparaison_graphique(resultats)

            afficher_diagnostics_modeles(
                y_test_reg,
                y_pred_reg,
                y_test_tree,
                y_pred_tree
            )

        else:
            print("Erreur : prétraitement impossible.")
    else:
        print("Erreur : chargement impossible.")