# -*- coding: utf-8 -*-
'''
Description multiple_regression.py
Projet : Prédiction de la note à partir des traces ARCHE

Module de régression multiple :
- sélection descendante des variables
- estimation du modèle linéaire
- évaluation sur train/test
- calcul du VIF

@author: Khady Diagne
@version: 1.1
@date: Avril 2026
'''

import pandas as pd
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

from config import TEST_SIZE, RANDOM_STATE, NOTES_TARGET_COL


def get_variables_candidates() -> list[str]:
    '''
    Retourne la liste des variables candidates pour la régression multiple.
    '''
    return [
        "nb_contextes",
        "ratio_test",
        "ratio_interaction",
        "temps_moyen_jour_actif",
        "temps_moyen_action",
        "ratio_fichier",
    ]


def selection_backward(
    df: pd.DataFrame,
    target: str = NOTES_TARGET_COL,
    seuil_pvalue: float = 0.05
):
    '''
    Sélection descendante des variables par p-value.

    :param df: dataframe final
    :param target: variable cible
    :param seuil_pvalue: seuil de significativité
    :return: variables retenues, modèle OLS final
    '''
    print("\nSÉLECTION DESCENDANTE DES VARIABLES...")

    variables = [var for var in get_variables_candidates() if var in df.columns]

    if len(variables) == 0:
        raise ValueError("Aucune variable candidate disponible pour la régression.")

    X = df[variables].copy()
    y = df[target].copy()

    while True:
        X_const = sm.add_constant(X)
        modele = sm.OLS(y, X_const).fit()

        pvalues = modele.pvalues.drop("const")
        max_p = pvalues.max()

        if max_p <= seuil_pvalue:
            break

        var_suppr = pvalues.idxmax()
        print(f"Suppression de {var_suppr} (p-value = {max_p:.4f})")
        X = X.drop(columns=[var_suppr])

        if X.shape[1] == 0:
            raise ValueError("Toutes les variables ont été supprimées par la sélection backward.")

    print("\nVariables retenues :")
    print(list(X.columns))

    print("\nRésumé du modèle final de sélection :")
    print(modele.summary())

    return list(X.columns), modele


def regression_multiple(
    df: pd.DataFrame,
    variables_retenues: list[str],
    target: str = NOTES_TARGET_COL
):
    '''
    Régression linéaire multiple.

    :param df: dataframe final
    :param variables_retenues: variables explicatives retenues
    :param target: variable cible
    :return: modèle, y_test, y_pred, rmse, r2
    '''
    print("\nLANCEMENT DE LA RÉGRESSION MULTIPLE...")

    X = df[variables_retenues].copy()
    y = df[target].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    modele = LinearRegression()
    modele.fit(X_train, y_train)

    y_pred = modele.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    print("\nÉvaluation sur l'échantillon de test :")
    print(f"MSE  : {mse:.3f}")
    print(f"RMSE : {rmse:.3f}")
    print(f"R²   : {r2:.3f}")

    print("\nCoefficients du modèle :")
    for nom, coef in zip(variables_retenues, modele.coef_):
        print(f"{nom:25} : {coef:.4f}")
    print(f"Intercept                 : {modele.intercept_:.4f}")

    return modele, y_test, y_pred, rmse, r2


def calcul_vif(df: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    '''
    Calcul du facteur d'inflation de variance.

    :param df: dataframe final
    :param variables: variables explicatives
    :return: tableau des VIF
    '''
    X = df[variables].copy()
    X = sm.add_constant(X)

    vif_data = pd.DataFrame()
    vif_data["variable"] = X.columns
    vif_data["VIF"] = [
        variance_inflation_factor(X.values, i)
        for i in range(X.shape[1])
    ]

    print("\nVIF (MULTICOLINÉARITÉ) :")
    print(vif_data)

    return vif_data


if __name__ == "__main__":
    print("TEST DU MODULE DE RÉGRESSION MULTIPLE...")

    from data_loader import load_data
    from preprocessing import preparer_donnees
    from features_engineering import construire_features

    df_logs, df_notes = load_data()

    if df_logs is not None and df_notes is not None:
        df_logs, df_notes = preparer_donnees(df_logs, df_notes)

        if df_logs is not None and df_notes is not None:
            df_final = construire_features(df_logs, df_notes)

            variables_retenues, modele_stats = selection_backward(df_final)
            modele, y_test, y_pred, rmse, r2 = regression_multiple(df_final, variables_retenues)
            vif_data = calcul_vif(df_final, variables_retenues)
        else:
            print("Erreur : prétraitement impossible.")
    else:
        print("Erreur : chargement impossible.")