# config.py

"""
Configuration générale du projet ARCHE.
Ce fichier centralise les constantes utilisées dans les différents modules.
"""
# Chemins des fichiers
LOGS_PATH = "data/logs.csv"
NOTES_PATH = "data/notes.csv"

# Colonnes du fichier logs
LOGS_TIME_COL = "heure"
LOGS_USER_COL = "pseudo"
LOGS_CONTEXT_COL = "contexte"
LOGS_COMPONENT_COL = "composant"
LOGS_EVENT_COL = "evenement"

# Colonnes du fichier notes
NOTES_USER_COL = "pseudo"
NOTES_TARGET_COL = "note"

# Colonnes attendues
EXPECTED_LOGS_COLUMNS = [
    LOGS_TIME_COL,
    LOGS_USER_COL,
    LOGS_CONTEXT_COL,
    LOGS_COMPONENT_COL,
    LOGS_EVENT_COL,
]

EXPECTED_NOTES_COLUMNS = [
    NOTES_USER_COL,
    NOTES_TARGET_COL,
]

# Paramètres du projet
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Modèle comparatif
COMPARISON_MODEL_NAME = "decision_tree_regressor"

# Paramètres exploration
TOP_N_EVENTS = 5

# Paramètres affichage
APP_TITLE = "Analyse des traces ARCHE"
WINDOW_SIZE = "700x500"