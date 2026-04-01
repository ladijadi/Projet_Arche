import pandas as pd


def construire_features(df_logs):
    "
    D = {}

    for _, row in df_logs.iterrows():
        pseudo = row["pseudo"]
        heure = pd.to_datetime(row["heure"], errors="coerce")
        contexte = str(row["contexte"])
        composant = str(row["composant"])
        evenement = str(row["evenement"]).lower()

        if pd.isna(heure):
            continue

        jour = heure.date()

        if pseudo not in D:
            D[pseudo] = {
                "nb_actions": 0,
                "jours": set(),
                "contextes": set(),
                "composants": set(),
                "heure_min": heure,
                "heure_max": heure,
                "nb_tests": 0,
            }

        D[pseudo]["nb_actions"] += 1
        D[pseudo]["jours"].add(jour)
        D[pseudo]["contextes"].add(contexte)
        D[pseudo]["composants"].add(composant)

        if heure < D[pseudo]["heure_min"]:
            D[pseudo]["heure_min"] = heure

        if heure > D[pseudo]["heure_max"]:
            D[pseudo]["heure_max"] = heure

        if "test" in evenement or "quiz" in evenement:
            D[pseudo]["nb_tests"] += 1

    data = []

    for pseudo, val in D.items():
        duree = (val["heure_max"] - val["heure_min"]).total_seconds()

        ratio = val["nb_tests"] / val["nb_actions"] if val["nb_actions"] > 0 else 0

        data.append({
            "pseudo": pseudo,
            "nb_actions": val["nb_actions"],
            "nb_jours_actifs": len(val["jours"]),
            "nb_contextes": len(val["contextes"]),
            "nb_composants": len(val["composants"]),
            "duree_totale": duree,
            "nb_tests": val["nb_tests"],
            "ratio_tests": ratio
        })

    return pd.DataFrame(data)