# config.py

"""
Fichier de configuration du projet ARCHE.
Il centralise les chemins des fichiers, les noms des colonnes
et quelques paramètres globaux.
"""

# Chemins des données
CHEMIN_LOGS = "data/logs.csv"
CHEMIN_NOTES = "data/notes.csv"

# Colonnes attendues dans logs.csv
COLONNE_HEURE = "heure"
COLONNE_PSEUDO = "pseudo"
COLONNE_CONTEXTE = "contexte"
COLONNE_COMPOSANT = "composant"
COLONNE_EVENEMENT = "evenement"

# Colonnes attendues dans notes.csv
COLONNE_NOTE = "note"

# Liste des colonnes obligatoires
COLONNES_LOGS_ATTENDUES = [
    COLONNE_HEURE,
    COLONNE_PSEUDO,
    COLONNE_CONTEXTE,
    COLONNE_COMPOSANT,
    COLONNE_EVENEMENT,
]

COLONNES_NOTES_ATTENDUES = [
    COLONNE_PSEUDO,
    COLONNE_NOTE,
]

# Paramètres de séparation apprentissage / test
TAILLE_TEST = 0.2
GRAINE_ALEATOIRE = 42

# Nom du modèle comparatif retenu
MODELE_COMPARATIF = "arbre_de_decision_regression"