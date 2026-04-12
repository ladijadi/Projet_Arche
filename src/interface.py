# -*- coding: utf-8 -*-
'''
Description interface.py
Projet : Prédiction de la note à partir des traces ARCHE

Module d'interface graphique :
- saisie d'un profil étudiant
- prédiction par deux modèles
- affichage d'une interprétation simple

@author: Khady Diagne
@version: 2.0
@date: Avril 2026
'''

import tkinter as tk
from tkinter import messagebox

import config

from data_loader import load_data
from preprocessing import preparer_donnees
from features_engineering import construire_features
from multiple_regression import selection_backward, regression_multiple
from comparison_model import arbre_decision_regression

class ApplicationPrediction(tk.Tk):
    '''
    Fenêtre principale de l'application
    '''

    def __init__(self):
        '''
        Constructeur
        '''
        super().__init__()

        self.title(config.APP_TITLE)
        self.geometry("720x650")
        self.minsize(680, 520)
        self.entries_variables = {}
        self.texte_reperes = "Chargement des repères..."
        
        self.modele_reg = None
        self.modele_tree = None
        self.variables_reg = None
        self.variables_tree = None

        self.rmse_reg = None
        self.r2_reg = None
        self.rmse_tree = None
        self.r2_tree = None

        self.df_final = None
        self.moyenne_nb_contextes = None
        self.moyenne_ratio_fichier = None

        self._creer_zone_scrollable()
        self._entrainer_modeles()
        self._creer_widgets()
        
        

    def _creer_zone_scrollable(self):
        '''
        Création d'une zone scrollable verticale
        '''
        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.scrollbar = tk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.container = tk.Frame(self.canvas)

        self.container.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas_window = self.canvas.create_window(
            (0, 0),
            window=self.container,
            anchor="nw"
        )

        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        self.canvas.bind(
            "<Configure>",
            lambda e: self.canvas.itemconfig(self.canvas_window, width=e.width)
        )

        self.bind_all("<MouseWheel>", self._on_mousewheel)

    def _on_mousewheel(self, event):
        '''
        Gestion de la molette
        '''
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _creer_widgets(self):
        '''
        Création des éléments graphiques
        '''
        parent = self.container

        tk.Label(
            parent,
            text="Prédiction de la note à partir des traces ARCHE",
            font=("Arial", 15, "bold")
        ).pack(pady=12)

        tk.Label(
            parent,
            text="Saisissez les indicateurs du profil étudiant puis lancez la prédiction.",
            font=("Arial", 10),
            fg="#444444"
        ).pack(pady=(0, 8))

        self.cadre_form = tk.Frame(parent)
        self.cadre_form.pack(pady=8)

        self._creer_champs_dynamiques(self.cadre_form)

        tk.Button(
            parent,
            text="Prédire la note",
            command=self.predire_note,
            width=22,
            bg="#DDEEFF",
            font=("Arial", 11, "bold")
        ).pack(pady=12)

        self._creer_bloc_resultat(parent)
        self._creer_bloc_modeles(parent)
        self._creer_bloc_interpretation(parent)
        self._creer_bloc_reperes(parent)

    def _creer_champs_dynamiques(self, parent):
        '''
        Création dynamique des champs de saisie selon les variables retenues
        '''
        self.entries_variables = {}

        libelles = {
            "nb_contextes": "Nombre de contextes consultés :",
            "ratio_test": "Ratio d'activités de test (0 à 1) :",
            "ratio_interaction": "Ratio d'interactions (0 à 1) :",
            "temps_moyen_action": "Temps moyen par action (en secondes) :",
            "ratio_fichier": "Ratio d'interactions avec les fichiers (0 à 1) :"
        }

        for i, variable in enumerate(self.variables_reg):
            texte = libelles.get(variable, f"{variable} :")

            tk.Label(
                parent,
                text=texte,
                font=("Arial", 11)
            ).grid(row=i, column=0, sticky="w", padx=8, pady=6)

            entry = tk.Entry(parent, width=20, font=("Arial", 11))
            entry.grid(row=i, column=1, padx=8, pady=6)

            self.entries_variables[variable] = entry

    def _creer_bloc_resultat(self, parent):
        cadre = tk.Frame(parent, bd=1, relief="solid", padx=12, pady=10)
        cadre.pack(pady=8, fill="x", padx=16)

        tk.Label(
            cadre,
            text="Résultat principal",
            font=("Arial", 12, "bold")
        ).pack(anchor="w")

        self.label_modele_recommande = tk.Label(
            cadre,
            text="Modèle recommandé : -",
            font=("Arial", 11, "bold"),
            fg="#003366"
        )
        self.label_modele_recommande.pack(anchor="w", pady=(6, 2))

        self.label_justification_modele = tk.Label(
            cadre,
            text="Justification : -",
            font=("Arial", 10),
            fg="#444444",
            wraplength=620,
            justify="left"
        )
        self.label_justification_modele.pack(anchor="w", pady=(0, 5))

        self.label_note_recommandee = tk.Label(
            cadre,
            text="Note estimée : -",
            font=("Arial", 14, "bold"),
            fg="#006600"
        )
        self.label_note_recommandee.pack(anchor="w", pady=(2, 3))

        self.label_incertitude = tk.Label(
            cadre,
            text="Précision estimée : ±1 point de note environ",
            font=("Arial", 9),
            fg="#666666"
        )
        self.label_incertitude.pack(anchor="w", pady=(0, 3))

        self.label_niveau = tk.Label(
            cadre,
            text="Niveau estimé : -",
            font=("Arial", 11, "bold"),
            fg="#7A3E00"
        )
        self.label_niveau.pack(anchor="w", pady=(2, 2))

        self.label_profil = tk.Label(
            cadre,
            text="Profil : -",
            font=("Arial", 11, "bold"),
            fg="#444444"
        )
        self.label_profil.pack(anchor="w", pady=(2, 0))

    def _creer_bloc_modeles(self, parent):
        cadre = tk.Frame(parent, bd=1, relief="solid", padx=12, pady=10)
        cadre.pack(pady=8, fill="x", padx=16)

        tk.Label(
            cadre,
            text="Détail des prédictions",
            font=("Arial", 12, "bold")
        ).pack(anchor="w")

        self.label_resultat_reg = tk.Label(
            cadre,
            text="Régression multiple : -",
            font=("Arial", 11),
            wraplength=620,
            justify="left"
        )
        self.label_resultat_reg.pack(anchor="w", pady=3)

        self.label_resultat_tree = tk.Label(
            cadre,
            text="Arbre de décision : -",
            font=("Arial", 11),
            wraplength=620,
            justify="left"
        )
        self.label_resultat_tree.pack(anchor="w", pady=3)

    def _creer_bloc_interpretation(self, parent):
        cadre = tk.Frame(parent, bd=1, relief="solid", padx=12, pady=10)
        cadre.pack(pady=8, fill="x", padx=16)

        tk.Label(
            cadre,
            text="Lecture du profil",
            font=("Arial", 12, "bold")
        ).pack(anchor="w")

        self.label_interpretation = tk.Label(
            cadre,
            text="Interprétation : -",
            wraplength=620,
            justify="left",
            font=("Arial", 10)
        )
        self.label_interpretation.pack(anchor="w", pady=(6, 4))

        self.label_recommandation = tk.Label(
            cadre,
            text="Recommandation : -",
            wraplength=620,
            justify="left",
            font=("Arial", 10, "bold"),
            fg="#7A3E00"
        )
        self.label_recommandation.pack(anchor="w", pady=(4, 0))

    def _creer_bloc_reperes(self, parent):
        cadre = tk.Frame(parent, bd=1, relief="solid", padx=12, pady=10)
        cadre.pack(pady=8, fill="x", padx=16)

        tk.Label(
            cadre,
            text="Repères du jeu de données",
            font=("Arial", 12, "bold")
        ).pack(anchor="w")

        self.label_reperes = tk.Label(
            cadre,
            text=self.texte_reperes,
            justify="left",
            font=("Arial", 10),
            wraplength=620
        )
        self.label_reperes.pack(anchor="w", pady=(6, 0))

    def _entrainer_modeles(self):
        '''
        Chargement des données et entraînement des modèles
        '''
        try:
            df_logs, df_notes = load_data()

            if df_logs is None or df_notes is None:
                raise ValueError("Chargement des données impossible.")

            df_logs, df_notes = preparer_donnees(df_logs, df_notes)

            if df_logs is None or df_notes is None:
                raise ValueError("Prétraitement des données impossible.")

            self.df_final = construire_features(df_logs, df_notes)

            self.moyenne_nb_contextes = self.df_final["nb_contextes"].mean()
            self.moyenne_ratio_fichier = self.df_final["ratio_fichier"].mean()

            # 1. Sélection des variables
            self.variables_reg, _ = selection_backward(self.df_final)

            if not isinstance(self.variables_reg, list) or len(self.variables_reg) == 0:
                raise ValueError(f"Variables de régression invalides : {self.variables_reg}")

            # 2. Texte des repères
            self.texte_reperes = (
                f"Nb contextes moyen : {self.moyenne_nb_contextes:.2f}\n"
                f"Ratio fichiers moyen : {self.moyenne_ratio_fichier:.2f}\n"
                f"Variables retenues : {', '.join(str(var) for var in self.variables_reg)}"
            )

            # 3. Entraînement de la régression
            self.modele_reg, _, _, self.rmse_reg, self.r2_reg = regression_multiple(
                self.df_final,
                self.variables_reg
            )

            # 4. Entraînement de l'arbre
            (
                self.modele_tree,
                _,
                _,
                _,
                self.rmse_tree,
                self.r2_tree,
                _,
                self.variables_tree
            ) = arbre_decision_regression(
                self.df_final,
                variables=self.variables_reg
            )

        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible d'initialiser les modèles :\n{e}")

    def predire_note(self):
        '''
        Prédiction de la note à partir des saisies utilisateur
        '''
        try:
            x_input = {}

            for variable, entry in self.entries_variables.items():
                valeur = float(entry.get().strip())

                if "ratio" in variable or "engagement" in variable:
                    if not (0 <= valeur <= 1):
                        raise ValueError(f"La variable {variable} doit être comprise entre 0 et 1.")

                if variable in ["nb_contextes", "temps_moyen_action"] and valeur < 0:
                    raise ValueError(f"La variable {variable} doit être positive.")

                x_input[variable] = valeur

            x_reg = [[x_input[var] for var in self.variables_reg]]
            prediction_reg = self.modele_reg.predict(x_reg)[0]

            x_tree = [[x_input[var] for var in self.variables_tree]]
            prediction_tree = self.modele_tree.predict(x_tree)[0]

            self.label_resultat_reg.config(
                text=(
                    f"Régression multiple : note prédite = {prediction_reg:.2f}/20 "
                    f"(RMSE = {self.rmse_reg:.3f}, R² = {self.r2_reg:.3f})"
                )
            )

            self.label_resultat_tree.config(
                text=(
                    f"Arbre de décision : note prédite = {prediction_tree:.2f}/20 "
                    f"(RMSE = {self.rmse_tree:.3f}, R² = {self.r2_tree:.3f})"
                )
            )

            modele_recommande, note_recommandee, justification = self._choisir_modele(
                prediction_reg,
                prediction_tree
            )

            self.label_modele_recommande.config(
                text=f"Modèle recommandé : {modele_recommande}"
            )

            self.label_justification_modele.config(
                text=f"Justification : {justification}"
            )

            self.label_note_recommandee.config(
                text=f"Note estimée : {note_recommandee:.2f}/20"
            )

            nb_contextes = x_input.get("nb_contextes", self.moyenne_nb_contextes)
            ratio_fichier = x_input.get("ratio_fichier", self.moyenne_ratio_fichier)

            self.label_niveau.config(
                text=f"Niveau estimé : {self._generer_niveau(note_recommandee)}"
            )

            self.label_profil.config(
                text=f"Profil : {self._classifier_profil(nb_contextes, ratio_fichier)}"
            )

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

    def _choisir_modele(self, pred_reg, pred_tree):
        '''
        Choisit le modèle à recommander à partir des performances globales.
        '''
        if self.rmse_tree < self.rmse_reg and self.r2_tree > self.r2_reg:
            justification = (
                "l'arbre de décision présente une erreur moyenne plus faible "
                "et un coefficient de détermination légèrement supérieur. "
                "Le gain reste toutefois modéré."
            )
            return "Arbre de décision régressif", pred_tree, justification

        if self.rmse_reg < self.rmse_tree and self.r2_reg > self.r2_tree:
            justification = (
                "la régression multiple présente de meilleures performances globales "
                "sur les métriques principales tout en restant plus interprétable."
            )
            return "Régression multiple", pred_reg, justification

        justification = (
            "les performances des deux modèles sont proches. "
            "La régression multiple reste plus lisible, tandis que l'arbre "
            "peut mieux représenter des effets de seuil."
        )
        return "Comparaison nuancée", pred_reg, justification

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
        commentaires = []

        if nb_contextes >= config.SEUIL_CONTEXTES_ELEVE:
            commentaires.append("activité diversifiée, associée à de meilleures performances")
        elif nb_contextes >= config.SEUIL_CONTEXTES_MOYEN:
            commentaires.append("activité modérée, impact limité sur la performance")
        else:
            commentaires.append("activité peu diversifiée, pouvant être associée à une performance plus faible")

        if ratio_fichier > config.SEUIL_RATIO_FICHIER_ELEVE:
            commentaires.append("forte dépendance aux fichiers, souvent associée à des notes plus faibles")
        elif ratio_fichier > config.SEUIL_RATIO_FICHIER_MOYEN:
            commentaires.append("usage modéré des fichiers")
        else:
            commentaires.append("profil plus équilibré dans l'utilisation des ressources")

        return " ; ".join(commentaires)

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

    def _generer_recommandation(self, nb_contextes, ratio_fichier):
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