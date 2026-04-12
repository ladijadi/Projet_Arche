# -*- coding: utf-8 -*-
'''
Description comparison_model.py
Projet : Prédiction de la note à partir des traces ARCHE

Module de modèle comparatif :
- entraînement d'un arbre de décision régressif
- évaluation sur échantillon de test
- affichage de l'arbre
- affichage de l'importance des variables

Remarque méthodologique :
l'arbre doit être entraîné sur le même espace de variables
que la régression multiple, afin de garantir une comparaison cohérente.

@author: Khady Diagne
@version: 2.0
@date: Avril 2026
'''

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from config import TEST_SIZE, RANDOM_STATE


def get_variables_comparaison() -> list[str]:
    '''
    Retourne la liste par défaut des variables candidates
    pour le modèle comparatif.

    :return: liste de variables explicatives
    '''
    return [
        "nb_contextes",
        "ratio_test",
        "ratio_interaction",
        "temps_moyen_action",
        "ratio_fichier",
    ]


def verifier_variables(df: pd.DataFrame, variables: list[str]) -> list[str]:
    '''
    Vérifie que les variables demandées existent dans le dataframe.

    :param df: dataframe final
    :param variables: liste des variables à contrôler
    :return: liste des variables présentes
    '''
    variables_presentes = [var for var in variables if var in df.columns]

    if len(variables_presentes) == 0:
        raise ValueError("Aucune variable valide n'a été trouvée pour l'arbre de décision.")

    return variables_presentes


def arbre_decision_regression(
    df: pd.DataFrame,
    variables: list[str] | None = None,
    target: str = "note",
    max_depth: int = 3,
    min_samples_leaf: int = 5
):
    '''
    Entraîne un arbre de décision régressif sur la note.

    :param df: dataframe final
    :param variables: variables explicatives utilisées
    :param target: variable cible
    :param max_depth: profondeur maximale de l'arbre
    :param min_samples_leaf: nombre minimum d'observations par feuille
    :return: modèle, X_test, y_test, y_pred, rmse, r2, mae, variables_utilisees
    '''
    print("\nLANCEMENT DU MODÈLE COMPARATIF : ARBRE DE DÉCISION RÉGRESSIF...")

    if variables is None:
        variables = get_variables_comparaison()

    variables_utilisees = verifier_variables(df, variables)

    print("\nVariables utilisées pour l'arbre :")
    print(variables_utilisees)

    X = df[variables_utilisees].copy()
    y = df[target].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    modele = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=RANDOM_STATE
    )

    modele.fit(X_train, y_train)

    y_pred = modele.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\nÉVALUATION DE L'ARBRE DE DÉCISION (TEST)")
    print(f"MSE  : {mse:.3f}")
    print(f"RMSE : {rmse:.3f}")
    print(f"MAE  : {mae:.3f}")
    print(f"R²   : {r2:.3f}")

    return modele, X_test, y_test, y_pred, rmse, r2, mae, variables_utilisees


def afficher_arbre_regression(modele, variables: list[str]) -> None:
    '''
    Affiche l'arbre de décision régressif.

    :param modele: arbre entraîné
    :param variables: liste des variables explicatives
    '''
    plt.figure(figsize=(16, 9))

    plot_tree(
        modele,
        feature_names=variables,
        filled=True,
        rounded=True,
        fontsize=9
    )

    plt.title("Arbre de décision régressif - prédiction de la note")
    plt.tight_layout()
    plt.show()


def importance_variables(modele, variables: list[str]) -> pd.DataFrame:
    '''
    Calcule et affiche l'importance des variables de l'arbre.

    :param modele: arbre entraîné
    :param variables: liste des variables explicatives
    :return: dataframe trié des importances
    '''
    importance = pd.DataFrame({
        "variable": variables,
        "importance": modele.feature_importances_
    })

    importance = importance.sort_values(by="importance", ascending=False)

    print("\nIMPORTANCE DES VARIABLES")
    print(importance.to_string(index=False))

    plt.figure(figsize=(8, 5))
    plt.bar(importance["variable"], importance["importance"])
    plt.title("Importance des variables - arbre de décision")
    plt.xlabel("Variables")
    plt.ylabel("Importance")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    return importance


if __name__ == "__main__":
    from data_loader import load_data
    from preprocessing import preparer_donnees
    from features_engineering import construire_features
    from multiple_regression import selection_backward

    print("TEST DU MODULE COMPARISON_MODEL")

    df_logs, df_notes = load_data()

    if df_logs is not None and df_notes is not None:
        df_logs, df_notes = preparer_donnees(df_logs, df_notes)

        if df_logs is not None and df_notes is not None:
            df_final = construire_features(df_logs, df_notes)

            variables_retenues, _ = selection_backward(df_final)

            (
                modele,
                X_test,
                y_test,
                y_pred,
                rmse,
                r2,
                mae,
                variables_utilisees
            ) = arbre_decision_regression(
                df=df_final,
                variables=variables_retenues
            )

            afficher_arbre_regression(modele, variables_utilisees)
            importance_variables(modele, variables_utilisees)

        else:
            print("Erreur : prétraitement impossible.")
    else:
        print("Erreur : chargement impossible.")