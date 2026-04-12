# Prédiction de la note à partir des traces sur ARCHE

Projet réalisé dans le cadre du Master 2 Science des Données (IDMC - Université de Lorraine, 2025-2026).

## Objectif

Ce projet vise à prédire la note finale d'un étudiant à partir de ses traces d'activité sur la plateforme pédagogique ARCHE.

Le projet repose sur deux sources de données :

- logs.csv : traces d'activité sur la plateforme
- notes.csv : notes finales des étudiants

Deux approches de modélisation sont comparées :

- Régression multiple (modèle imposé)
- Arbre de décision régressif (modèle comparatif)

Une interface graphique développée avec Tkinter permet de tester les modèles à partir d'un profil étudiant saisi manuellement.

## Structure du projet

```text
Projet_Arche/
├── data/
│   ├── logs.csv
│   └── notes.csv
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── exploration.py
│   ├── features_engineering.py
│   ├── multiple_regression.py
│   ├── comparison_model.py
│   ├── evaluation.py
│   └── interface.py
├── output/
├── main.py
└── README.md
```

## Étapes du pipeline

Le projet suit les étapes suivantes :

1. Chargement et validation des données
2. Prétraitement des logs et des notes
3. Exploration descriptive
4. Construction des variables explicatives
5. Modélisation par régression multiple
6. Modélisation comparative par arbre de décision
7. Évaluation des performances
8. Restitution via une interface graphique

## Variables construites

À partir des logs, plusieurs variables explicatives sont construites au niveau étudiant, notamment :

- nb_contextes
- nb_actions
- nb_jours_actifs
- temps_moyen_action
- temps_moyen_jour_actif
- ratio_test
- ratio_interaction
- ratio_consultation
- engagement_actif
- ratio_fichier

Le modèle final de régression multiple retient :

- nb_contextes
- ratio_fichier

## Résultats principaux

### Régression multiple

- Variables retenues : nb_contextes, ratio_fichier
- RMSE ≈ 0.910
- R² ≈ 0.233

### Arbre de décision régressif

- RMSE ≈ 0.870
- MAE ≈ 0.718
- R² ≈ 0.299

L'arbre de décision présente des performances légèrement supérieures, tandis que la régression multiple reste plus interprétable.

## Lancer le projet

Depuis la racine du projet :

```bash
python main.py
```

## Dépendances principales

Le projet utilise notamment :

- pandas
- numpy
- matplotlib
- scikit-learn
- statsmodels
- tkinter

Installation possible avec :

```bash
pip install pandas numpy matplotlib scikit-learn statsmodels
```

tkinter est généralement inclus avec Python selon l'environnement.

## Interface graphique

L'interface permet :

- de saisir les variables explicatives retenues
- d'obtenir la prédiction des deux modèles
- d'afficher un modèle recommandé
- de proposer une interprétation synthétique du profil
- de situer l'utilisateur par rapport aux repères du jeu de données

## Limites

Le projet présente plusieurs limites :

- les notes sont relativement peu dispersées
- les logs ne capturent qu'une partie du comportement étudiant
- certains facteurs importants ne sont pas observés (travail hors plateforme, niveau initial, motivation)
- les performances prédictives restent modérées

Ce projet doit être interprété comme une démonstration académique de modélisation à partir de traces numériques, et non comme un outil de décision opérationnel.

## Auteur

Khady Diagne  
Master 2 Science des Données  
IDMC - Université de Lorraine

## Livrables associés

- dossier d'analyse
- dossier technique
- code source
- soutenance orale

## Droits d'auteur

© 2026 Khady Diagne. Tous droits réservés.

Ce projet a été réalisé dans un cadre académique (Master 2 Science des Données, IDMC - Université de Lorraine). Le code source et les documents associés ne peuvent pas être reproduits, modifiés, distribués ou exploités à des fins commerciales sans autorisation écrite préalable de l'autrice.


