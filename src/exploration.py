# exploration.py

import pandas as pd
import matplotlib.pyplot as plt
from src.config import LOGS_USER_COL, LOGS_CONTEXT_COL, LOGS_EVENT_COL, NOTES_USER_COL, NOTES_TARGET_COL


def explorer_logs(df_logs: pd.DataFrame) -> None:
    print("\nEXPLORATION DES LOGS")

    nb_actions_total = len(df_logs)
    nb_etudiants_actifs = df_logs[LOGS_USER_COL].nunique()
    nb_contextes = df_logs[LOGS_CONTEXT_COL].nunique()

    print(f"Nombre total d'actions enregistrées : {nb_actions_total}")
    print(f"Nombre d'étudiants actifs dans les logs : {nb_etudiants_actifs}")
    print(f"Nombre de contextes distincts consultés : {nb_contextes}")

    print("\nTop 5 des types d'événements les plus fréquents :")
    print(df_logs[LOGS_EVENT_COL].value_counts().head(5).to_string())

    actions_par_etudiant = df_logs.groupby(LOGS_USER_COL).size()

    print("\nRésumé du nombre d'actions par étudiant :")
    print(f"Minimum : {actions_par_etudiant.min()}")
    print(f"Moyenne : {actions_par_etudiant.mean():.2f}")
    print(f"Maximum : {actions_par_etudiant.max()}")


def explorer_notes(df_notes: pd.DataFrame) -> None:
    print("\nEXPLORATION DES NOTES")

    nb_etudiants_notes = df_notes[NOTES_USER_COL].nunique()
    note_min = df_notes[NOTES_TARGET_COL].min()
    note_moyenne = df_notes[NOTES_TARGET_COL].mean()
    note_max = df_notes[NOTES_TARGET_COL].max()

    print(f"Nombre d'étudiants dans le fichier notes : {nb_etudiants_notes}")
    print(f"Note minimale : {note_min}")
    print(f"Note moyenne : {note_moyenne:.2f}")
    print(f"Note maximale : {note_max}")


def identifier_etudiants_sans_activite(df_logs: pd.DataFrame, df_notes: pd.DataFrame) -> pd.DataFrame:
    print("\nRECHERCHE DES ÉTUDIANTS SANS ACTIVITÉ SUR ARCHE")

    pseudos_actifs = set(df_logs[LOGS_USER_COL].unique())
    pseudos_notes = set(df_notes[NOTES_USER_COL].unique())

    pseudos_absents = pseudos_notes - pseudos_actifs
    df_absents = df_notes[df_notes[NOTES_USER_COL].isin(pseudos_absents)]

    print(f"Nombre d'étudiants sans activité sur ARCHE : {len(df_absents)}")

    if len(df_absents) > 0:
        print("Liste des étudiants sans activité :")
        print(df_absents[[NOTES_USER_COL, NOTES_TARGET_COL]].to_string(index=False))
    else:
        print("Tous les étudiants ayant une note possèdent au moins une activité.")

    return df_absents


def analyser_relation_simple(df_logs: pd.DataFrame, df_notes: pd.DataFrame) -> pd.DataFrame:
    print("\nPREMIÈRE ANALYSE LIEN ACTIVITÉ / NOTE")

    df_actions = df_logs.groupby(LOGS_USER_COL).size().reset_index(name="nb_actions")
    df_fusion = df_actions.merge(df_notes, left_on=LOGS_USER_COL, right_on=NOTES_USER_COL, how="inner")

    correlation = df_fusion["nb_actions"].corr(df_fusion[NOTES_TARGET_COL])

    print(f"Corrélation entre nombre d'actions et note : {correlation:.3f}")

    if pd.isna(correlation):
        print("La corrélation n'a pas pu être calculée.")
    elif abs(correlation) < 0.2:
        print("Le lien linéaire observé entre activité brute et note semble faible.")
    else:
        print("Un signal intéressant apparaît entre activité brute et note.")

    return df_fusion


def afficher_tous_les_graphiques(df_logs, df_notes, df_fusion):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Histogramme actions
    actions_par_etudiant = df_logs.groupby("pseudo").size()
    axes[0, 0].hist(actions_par_etudiant, bins=20)
    axes[0, 0].set_title("Actions par étudiant")

    # Histogramme notes
    axes[0, 1].hist(df_notes["note"], bins=15)
    axes[0, 1].set_title("Distribution des notes")

    # Top événements
    top_evenements = df_logs["evenement"].value_counts().head(5)
    axes[1, 0].bar(top_evenements.index, top_evenements.values)
    axes[1, 0].set_title("Top événements")
    axes[1, 0].tick_params(axis='x', rotation=20)

    # Scatter
    axes[1, 1].scatter(df_fusion["nb_actions"], df_fusion["note"])
    axes[1, 1].set_title("Actions vs Note")

    plt.tight_layout()
    plt.show()

def lancer_exploration(df_logs: pd.DataFrame, df_notes: pd.DataFrame) -> None:
    print("LANCEMENT DE L'EXPLORATION")
    
    explorer_logs(df_logs)
    explorer_notes(df_notes)
    identifier_etudiants_sans_activite(df_logs, df_notes)
    df_fusion = analyser_relation_simple(df_logs, df_notes)

    print("\nAFFICHAGE DES GRAPHIQUES")
    afficher_tous_les_graphiques(df_logs, df_notes, df_fusion)

    print("\nExploration terminée.")

if __name__ == "__main__":
    from src.data_loader import load_data
    from src.preprocessing import preparer_donnees

    df_logs, df_notes = load_data()

    if df_logs is not None and df_notes is not None:
        df_logs, df_notes = preparer_donnees(df_logs, df_notes)

        if df_logs is not None and df_notes is not None:
            lancer_exploration(df_logs, df_notes)