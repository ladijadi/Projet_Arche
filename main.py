from pathlib import Path
import sys

# Ajout du dossier src au chemin d'import
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from interface import ApplicationPrediction


def main():
    """
    Point d'entrée principal du projet.
    Lance directement l'interface graphique.
    """
    print("\nPROJET ARCHE - PRÉDICTION DE NOTES")
    print("Lancement de l'interface utilisateur...")

    app = ApplicationPrediction()
    app.mainloop()


if __name__ == "__main__":
    main()