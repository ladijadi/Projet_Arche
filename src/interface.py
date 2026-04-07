# interface.py

import config
import tkinter as tk
from tkinter import messagebox

from data_loader import load_data
from preprocessing import preparer_donnees
from features_engineering import construire_features
from multiple_regression import selection_backward, regression_multiple
from comparison_model import arbre_decision_regression


class ApplicationPrediction(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title(config.APP_TITLE)
        self.geometry(config.WINDOW_SIZE)
        self.resizable(False, True)

        self.modele_reg = None
        self.modele_tree = None
        self.variables_reg = None

        self.df_final = None
        self.moyenne_nb_contextes = None
        self.moyenne_ratio_fichier = None

        self._creer_widgets()
        self._entrainer_modeles()

    def _creer_widgets(self):
        # Titre
        tk.Label(
            self,
            text="Prédiction de la note à partir des traces ARCHE",
            font=("Arial", 16, "bold")
        ).pack(pady=15)

        tk.Label(
            self,
            text="Saisissez les indicateurs du profil étudiant puis lancez la prédiction.",
            font=("Arial", 10),
            fg="#444444"
        ).pack(pady=(0, 10))

        # Formulaire
        cadre_form = tk.Frame(self)
        cadre_form.pack(pady=10)

        tk.Label(
            cadre_form,
            text="Nombre de contextes consultés :",
            font=("Arial", 11)
        ).grid(row=0, column=0, sticky="w", padx=10, pady=8)

        self.entry_nb_contextes = tk.Entry(cadre_form, width=20, font=("Arial", 11))
        self.entry_nb_contextes.grid(row=0, column=1, padx=10, pady=8)

        tk.Label(
            cadre_form,
            text="Ratio d'interactions avec les fichiers (0 à 1) :",
            font=("Arial", 11)
        ).grid(row=1, column=0, sticky="w", padx=10, pady=8)

        self.entry_ratio_fichier = tk.Entry(cadre_form, width=20, font=("Arial", 11))
        self.entry_ratio_fichier.grid(row=1, column=1, padx=10, pady=8)

        tk.Button(
            self,
            text="Prédire la note",
            command=self.predire_note,
            width=22,
            bg="#DDEEFF",
            font=("Arial", 11, "bold")
        ).pack(pady=15)

        # Bloc résultat principal
        cadre_resultat = tk.Frame(self, bd=1, relief="solid", padx=15, pady=12)
        cadre_resultat.pack(pady=10, fill="x", padx=20)

        tk.Label(
            cadre_resultat,
            text="Résultat principal",
            font=("Arial", 12, "bold")
        ).pack(anchor="w")

        self.label_modele_recommande = tk.Label(
            cadre_resultat,
            text="Modèle recommandé : -",
            font=("Arial", 11, "bold"),
            fg="#003366"
        )
        self.label_modele_recommande.pack(anchor="w", pady=(8, 2))

        self.label_justification_modele = tk.Label(
            cadre_resultat,
            text="Justification : -",
            font=("Arial", 10),
            fg="#444444",
            wraplength=560,
            justify="left"
        )
        self.label_justification_modele.pack(anchor="w", pady=(0, 6))

        self.label_note_recommandee = tk.Label(
            cadre_resultat,
            text="Note estimée : -",
            font=("Arial", 15, "bold"),
            fg="#006600"
        )
        self.label_note_recommandee.pack(anchor="w", pady=(2, 4))

        self.label_incertitude = tk.Label(
            cadre_resultat,
            text="Précision estimée : ±1 point de note",
            font=("Arial", 9),
            fg="#666666"
        )
        self.label_incertitude.pack(anchor="w", pady=(0, 4))

        self.label_niveau = tk.Label(
            cadre_resultat,
            text="Niveau estimé : -",
            font=("Arial", 11, "bold"),
            fg="#7A3E00"
        )
        self.label_niveau.pack(anchor="w", pady=(2, 2))

        self.label_profil = tk.Label(
            cadre_resultat,
            text="Profil : -",
            font=("Arial", 11, "bold"),
            fg="#444444"
        )
        self.label_profil.pack(anchor="w", pady=(2, 0))

        # Bloc comparaison
        cadre_modeles = tk.Frame(self, bd=1, relief="solid", padx=15, pady=12)
        cadre_modeles.pack(pady=10, fill="x", padx=20)

        tk.Label(
            cadre_modeles,
            text="Détail des prédictions",
            font=("Arial", 12, "bold")
        ).pack(anchor="w")

        self.label_resultat_reg = tk.Label(
            cadre_modeles,
            text="Régression multiple : -",
            font=("Arial", 11)
        )
        self.label_resultat_reg.pack(anchor="w", pady=4)

        self.label_resultat_tree = tk.Label(
            cadre_modeles,
            text="Arbre de décision : -",
            font=("Arial", 11)
        )
        self.label_resultat_tree.pack(anchor="w", pady=4)

        # Bloc lecture / recommandation
        cadre_interpretation = tk.Frame(self, bd=1, relief="solid", padx=15, pady=12)
        cadre_interpretation.pack(pady=10, fill="x", padx=20)

        tk.Label(
            cadre_interpretation,
            text="Lecture du profil",
            font=("Arial", 12, "bold")
        ).pack(anchor="w")

        self.label_interpretation = tk.Label(
            cadre_interpretation,
            text="Interprétation : -",
            wraplength=560,
            justify="left",
            font=("Arial", 10)
        )
        self.label_interpretation.pack(anchor="w", pady=(8, 5))

        self.label_recommandation = tk.Label(
            cadre_interpretation,
            text="Recommandation : -",
            wraplength=560,
            justify="left",
            font=("Arial", 10, "bold"),
            fg="#7A3E00"
        )
        self.label_recommandation.pack(anchor="w", pady=(5, 0))

        # Bloc repères
        cadre_reperes = tk.Frame(self, bd=1, relief="solid", padx=15, pady=12)
        cadre_reperes.pack(pady=10, fill="x", padx=20)

        tk.Label(
            cadre_reperes,
            text="Repères du jeu de données",
            font=("Arial", 12, "bold")
        ).pack(anchor="w")

        self.label_reperes = tk.Label(
            cadre_reperes,
            text="Chargement des repères...",
            justify="left",
            font=("Arial", 10)
        )
        self.label_reperes.pack(anchor="w", pady=(8, 0))

    def _entrainer_modeles(self):
        """
        Charge les données et entraîne les deux modèles au démarrage.
        """
        try:
            df_logs, df_notes = load_data()

            if df_logs is None or df_notes is None:
                raise ValueError("Chargement des données impossible.")

            df_logs, df_notes = preparer_donnees(df_logs, df_notes)

            if df_logs is None or df_notes is None:
                raise ValueError("Prétraitement des données impossible.")

            df_final = construire_features(df_logs, df_notes)
            self.df_final = df_final

            # Repères moyens
            self.moyenne_nb_contextes = df_final["nb_contextes"].mean()
            self.moyenne_ratio_fichier = df_final["ratio_fichier"].mean()

            self.label_reperes.config(
                text=(
                    f"Nb contextes moyen : {self.moyenne_nb_contextes:.2f}\n"
                    f"Ratio fichiers moyen : {self.moyenne_ratio_fichier:.2f}"
                )
            )

            # Régression multiple
            self.variables_reg, _ = selection_backward(df_final)
            self.modele_reg, _, _, _, _ = regression_multiple(df_final, self.variables_reg)

            # Arbre régressif
            self.modele_tree, _, _, _, _, _ = arbre_decision_regression(df_final)

        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible d'initialiser les modèles :\n{e}")

    def predire_note(self):
        """
        Prédit la note avec les deux modèles à partir des variables finales.
        """
        try:
            nb_contextes = float(self.entry_nb_contextes.get().strip())
            ratio_fichier = float(self.entry_ratio_fichier.get().strip())

            if nb_contextes < 0:
                raise ValueError("Le nombre de contextes doit être positif.")

            if not (0 <= ratio_fichier <= 1):
                raise ValueError("Le ratio_fichier doit être compris entre 0 et 1.")

            x_input = {
                "nb_contextes": nb_contextes,
                "ratio_fichier": ratio_fichier
            }

            # Régression multiple
            x_reg = [[x_input[var] for var in self.variables_reg]]
            prediction_reg = self.modele_reg.predict(x_reg)[0]

            # Arbre de décision
            x_tree = [[nb_contextes, ratio_fichier]]
            prediction_tree = self.modele_tree.predict(x_tree)[0]

            # Détail des prédictions
            self.label_resultat_reg.config(
                text=f"Régression multiple : note prédite = {prediction_reg:.2f}/20"
            )

            self.label_resultat_tree.config(
                text=f"Arbre de décision : note prédite = {prediction_tree:.2f}/20"
            )

            # Modèle recommandé
            modele_recommande, note_recommandee = self._choisir_modele(prediction_reg, prediction_tree)

            self.label_modele_recommande.config(
                text=f"Modèle recommandé : {modele_recommande}"
            )

            self.label_justification_modele.config(
                text="Justification : modèle légèrement plus performant sur cet échantillon, "
                     "mais les performances globales restent modérées."
            )

            self.label_note_recommandee.config(
                text=f"Note estimée : {note_recommandee:.2f}/20"
            )

            self.label_niveau.config(
                text=f"Niveau estimé : {self._generer_niveau(note_recommandee)}"
            )

            self.label_profil.config(
                text=f"Profil : {self._classifier_profil(nb_contextes, ratio_fichier)}"
            )

            # Lecture + recommandation
            interpretation = self._generer_interpretation(nb_contextes, ratio_fichier)
            positionnement = self._generer_positionnement(nb_contextes, ratio_fichier)
            recommandation = self._generer_recommandation(nb_contextes, ratio_fichier)

            self.label_interpretation.config(
                text=f"Interprétation : {interpretation}\nPositionnement : {positionnement}"
            )

            self.label_recommandation.config(
                text=f"Recommandation : {recommandation}"
            )

        except ValueError as e:
            messagebox.showwarning("Entrée invalide", str(e))
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de faire la prédiction :\n{e}")

    def _generer_positionnement(self, nb_contextes, ratio_fichier):
        messages = []

        if nb_contextes < self.moyenne_nb_contextes:
            messages.append("activité moins diversifiée que la moyenne des étudiants")
        else:
            messages.append("activité aussi diversifiée ou plus diversifiée que la moyenne")

        if ratio_fichier < self.moyenne_ratio_fichier:
            messages.append("usage des fichiers inférieur à la moyenne")
        else:
            messages.append("usage des fichiers supérieur ou égal à la moyenne")

        return " ; ".join(messages)

    def _choisir_modele(self, pred_reg, pred_tree):
        """
        Retourne le modèle mis en avant dans l'interface.
        """
        if config.MODELE_RECOMMANDE == "Arbre de décision régressif":
            return config.MODELE_RECOMMANDE, pred_tree
        return config.MODELE_RECOMMANDE, pred_reg

    def _generer_niveau(self, note):
        if note >= 14:
            return "Bon niveau"
        elif note >= 10:
            return "Niveau moyen"
        return "Niveau fragile"

    def _classifier_profil(self, nb_contextes, ratio_fichier):
        if (
            nb_contextes >= self.moyenne_nb_contextes
            and ratio_fichier < self.moyenne_ratio_fichier
        ):
            return "Profil engagé et équilibré"
        elif (
            nb_contextes < self.moyenne_nb_contextes
            and ratio_fichier > self.moyenne_ratio_fichier
        ):
            return "Profil à risque (activité faible et déséquilibrée)"
        else:
            return "Profil intermédiaire"

    def _generer_interpretation(self, nb_contextes, ratio_fichier):
        """
        Donne une lecture simple du profil.
        """
        commentaires = []

        if nb_contextes >= config.SEUIL_CONTEXTES_ELEVE:
            commentaires.append("activité diversifiée, associée à de meilleures performances")
        elif nb_contextes >= config.SEUIL_CONTEXTES_MOYEN:
            commentaires.append("activité modérée, impact limité sur la performance")
        else:
            commentaires.append("activité peu diversifiée, pouvant expliquer une performance plus faible")

        if ratio_fichier > config.SEUIL_RATIO_FICHIER_ELEVE:
            commentaires.append("forte dépendance aux fichiers, associée à des notes plus faibles dans les données")
        elif ratio_fichier > config.SEUIL_RATIO_FICHIER_MOYEN:
            commentaires.append("usage modéré des fichiers")
        else:
            commentaires.append("profil équilibré dans l'utilisation des ressources")

        return " ; ".join(commentaires)

    def _generer_recommandation(self, nb_contextes, ratio_fichier):
        """
        Propose une recommandation simple et actionnable.
        """
        if nb_contextes < config.SEUIL_CONTEXTES_MOYEN and ratio_fichier > config.SEUIL_RATIO_FICHIER_ELEVE:
            return "Augmenter la diversité des ressources consultées et ne pas se limiter aux fichiers."
        elif nb_contextes < config.SEUIL_CONTEXTES_MOYEN:
            return "Explorer davantage de contextes et varier les contenus consultés."
        elif ratio_fichier > config.SEUIL_RATIO_FICHIER_ELEVE:
            return "Rééquilibrer l'activité en consultant aussi d'autres types de ressources que les fichiers."
        else:
            return (
                "Profil globalement cohérent. Continuer à varier les types de ressources "
                "(forums, tests, contenus) pour maintenir un bon équilibre d’apprentissage."
            )


if __name__ == "__main__":
    app = ApplicationPrediction()
    app.mainloop()