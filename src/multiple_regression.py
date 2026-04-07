# multiple_regression.py

import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def selection_backward(df, target="note", seuil_pvalue=0.05):
    """
    Sélection descendante des variables par p-value.
    On part d'un modèle complet raisonnable, puis on retire
    les variables non significatives une à une.
    """
    variables = [
        "nb_contextes",
        "ratio_test",
        "ratio_interaction",
        "temps_moyen_action",
        "ratio_fichier",
    ]

    X = df[variables].copy()
    y = df[target]

    while True:
        X_const = sm.add_constant(X)
        modele = sm.OLS(y, X_const).fit()

        pvalues = modele.pvalues.drop("const")
        max_p = pvalues.max()

        if max_p <= seuil_pvalue:
            break

        variable_a_supprimer = pvalues.idxmax()
        print(f"Suppression de {variable_a_supprimer} (p-value = {max_p:.4f})")
        X = X.drop(columns=[variable_a_supprimer])

    print("\nVariables retenues :")
    print(list(X.columns))

    print("\nRésumé du modèle final de sélection :")
    print(modele.summary())

    return list(X.columns), modele

def regression_multiple(df, variables_retenues, target="note"):
    """
    Entraîne la régression multiple finale avec sklearn
    pour évaluer le modèle sur train/test.
    """
    print("\nLancement de la régression multiple...")

    X = df[variables_retenues]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
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
        print(f"{nom:20} : {coef:.4f}")
    print(f"Intercept            : {modele.intercept_:.4f}")

    return modele, y_test, y_pred, rmse, r2

if __name__ == "__main__":
    from data_loader import load_data
    from preprocessing import preparer_donnees
    from features_engineering import construire_features

    print("Exécution régression multiple...")

    df_logs, df_notes = load_data()

    if df_logs is not None and df_notes is not None:
        df_logs, df_notes = preparer_donnees(df_logs, df_notes)

        if df_logs is not None and df_notes is not None:
            df_final = construire_features(df_logs, df_notes)

            variables_retenues, modele_stats = selection_backward(df_final)
            regression_multiple(df_final, variables_retenues)
        else:
            print("Erreur : prétraitement impossible.")
    else:
        print("Erreur : chargement impossible.")