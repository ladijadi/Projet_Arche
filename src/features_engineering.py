# features_engineering.py

import numpy as np
import pandas as pd

from config import (
    LOGS_USER_COL,
    LOGS_TIME_COL,
    LOGS_CONTEXT_COL,
    LOGS_COMPONENT_COL,
    NOTES_USER_COL,
)


def construire_features(df_logs: pd.DataFrame, df_notes: pd.DataFrame) -> pd.DataFrame:

    df_logs = df_logs.copy()

    print("\nConstruction des features finales...")

    # =========================
    # 1. FEATURES STRUCTURELLES
    # =========================

    # Diversité (clé)
    nb_contextes = (
        df_logs.groupby(LOGS_USER_COL)[LOGS_CONTEXT_COL]
        .nunique()
        .reset_index(name="nb_contextes")
    )

    # Nombre d’actions
    nb_actions = (
        df_logs.groupby(LOGS_USER_COL)
        .size()
        .reset_index(name="nb_actions")
    )

    # Nombre de jours actifs (IMPORTANT)
    df_logs["jour"] = df_logs[LOGS_TIME_COL].dt.date
    nb_jours = (
        df_logs.groupby(LOGS_USER_COL)["jour"]
        .nunique()
        .reset_index(name="nb_jours_actifs")
    )

    # =========================
    # 2. TEMPS
    # =========================

    df_logs = df_logs.sort_values(by=[LOGS_USER_COL, LOGS_TIME_COL])

    df_logs["ecart"] = (
        df_logs.groupby(LOGS_USER_COL)[LOGS_TIME_COL]
        .diff()
        .dt.total_seconds()
    )

    petits_ecarts = df_logs[
        (df_logs["ecart"] > 0) & (df_logs["ecart"] < 600)
    ]

    temps_total = (
        petits_ecarts.groupby(LOGS_USER_COL)["ecart"]
        .sum()
        .reset_index(name="temps_total_sec")
    )

    df_temps = nb_actions.merge(temps_total, on=LOGS_USER_COL, how="left")

    df_temps["temps_moyen_action"] = np.where(
        df_temps["nb_actions"] > 0,
        df_temps["temps_total_sec"] / df_temps["nb_actions"],
        0
    )

    df_temps = df_temps[[LOGS_USER_COL, "temps_moyen_action"]]

    # =========================
    # 3. COMPORTEMENT PAR TYPE (UTILISE categorie_evenement)
    # =========================

    # Comptage par catégorie
    df_cat = (
        df_logs.groupby([LOGS_USER_COL, "categorie_evenement"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    # On sécurise les colonnes
    for col in ["test", "interaction", "consultation"]:
        if col not in df_cat.columns:
            df_cat[col] = 0

    # Merge avec nb_actions pour ratios
    df_cat = df_cat.merge(nb_actions, on=LOGS_USER_COL, how="left")

    # Ratios comportementaux
    df_cat["ratio_test"] = df_cat["test"] / df_cat["nb_actions"]
    df_cat["ratio_interaction"] = df_cat["interaction"] / df_cat["nb_actions"]
    df_cat["ratio_consultation"] = df_cat["consultation"] / df_cat["nb_actions"]

    # Engagement global (TRÈS IMPORTANT)
    df_cat["engagement_actif"] = (
        df_cat["test"] + df_cat["interaction"]
    ) / df_cat["nb_actions"]

    df_cat = df_cat[
        [
            LOGS_USER_COL,
            "ratio_test",
            "ratio_interaction",
            "ratio_consultation",
            "engagement_actif",
        ]
    ]

    # =========================
    # 4. FEATURES COMPOSANT
    # =========================

    nb_fichier = (
        df_logs[df_logs[LOGS_COMPONENT_COL].str.lower() == "fichier"]
        .groupby(LOGS_USER_COL)
        .size()
        .reset_index(name="nb_fichier")
    )

    df_ratio = nb_actions.merge(nb_fichier, on=LOGS_USER_COL, how="left")
    df_ratio["nb_fichier"] = df_ratio["nb_fichier"].fillna(0)

    df_ratio["ratio_fichier"] = df_ratio["nb_fichier"] / df_ratio["nb_actions"]

    df_ratio = df_ratio[[LOGS_USER_COL, "ratio_fichier"]]

    # =========================
    # 5. MERGE FINAL
    # =========================

    df_final = df_notes.copy()

    features = [
        nb_contextes,
        nb_actions,
        nb_jours,
        df_temps,
        df_cat,
        df_ratio,
    ]

    for df_feat in features:
        df_final = df_final.merge(df_feat, on=NOTES_USER_COL, how="left")

    df_final = df_final.fillna(0)

    print(f"Features finales construites : {len(df_final)} étudiants")

    return df_final

if __name__ == "__main__": 
    print("Test du module features") 
    from data_loader import load_data 
    from preprocessing import preparer_donnees 

    df_logs, df_notes = load_data() 
    
    if df_logs is not None and df_notes is not None: 
        df_logs, df_notes = preparer_donnees(df_logs, df_notes) 
        
        if df_logs is not None and df_notes is not None:
            df_final = construire_features(df_logs, df_notes)
            print("\nAperçu rapide :")
            print(df_final.head().to_string(index=False))
            print("\nColonnes du jeu final :")
            print(df_final.columns.tolist())
        else:
            print("Erreur : prétraitement impossible.")
    else:
        print("Erreur : chargement impossible.")
