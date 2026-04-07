# comparison_model.py

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, r2_score


def arbre_decision_regression(df: pd.DataFrame):
    """
    Modèle comparatif : arbre de décision régressif sur la note.
    On compare ainsi deux modèles sur la même cible.
    """
    print("\nLancement du modèle comparatif : arbre de décision régressif...")

    variables = ["nb_contextes", "ratio_fichier"]
    X = df[variables]
    y = df["note"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    modele = DecisionTreeRegressor(
        max_depth=3,
        min_samples_leaf=5,
        random_state=42
    )
    modele.fit(X_train, y_train)

    y_pred = modele.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    print("\nÉvaluation de l'arbre de décision (test) :")
    print(f"MSE  : {mse:.3f}")
    print(f"RMSE : {rmse:.3f}")
    print(f"R²   : {r2:.3f}")

    return modele, X_test, y_test, y_pred, rmse, r2


def afficher_arbre_regression(modele, variables):
    """
    Affiche l'arbre de décision régressif.
    """
    plt.figure(figsize=(14, 8))
    plot_tree(
        modele,
        feature_names=variables,
        filled=True,
        rounded=True,
        fontsize=10
    )
    plt.title("Arbre de décision régressif - prédiction de la note")
    plt.tight_layout()
    plt.show()


def importance_variables(modele, variables):
    """
    Affiche l'importance des variables de l'arbre.
    """
    importance = pd.DataFrame({
        "variable": variables,
        "importance": modele.feature_importances_
    }).sort_values(by="importance", ascending=False)

    print("\nImportance des variables :")
    print(importance.to_string(index=False))

    plt.figure(figsize=(8, 5))
    plt.bar(importance["variable"], importance["importance"])
    plt.xticks(rotation=45)
    plt.title("Importance des variables - arbre de décision")
    plt.tight_layout()
    plt.show()

    return importance


if __name__ == "__main__":
    from data_loader import load_data
    from preprocessing import preparer_donnees
    from features_engineering import construire_features

    print("Exécution arbre de décision régressif...")

    df_logs, df_notes = load_data()

    if df_logs is not None and df_notes is not None:
        df_logs, df_notes = preparer_donnees(df_logs, df_notes)

        if df_logs is not None and df_notes is not None:
            df_final = construire_features(df_logs, df_notes)

            modele, X_test, y_test, y_pred, rmse, r2 = arbre_decision_regression(df_final)
            afficher_arbre_regression(modele, ["nb_contextes", "ratio_fichier"])
            importance_variables(modele, ["nb_contextes", "ratio_fichier"])
        else:
            print("Erreur : prétraitement impossible.")
    else:
        print("Erreur : chargement impossible.")