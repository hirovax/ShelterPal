"""
Finalny model CatBoost: czas pobytu w schronisku (<=4 dni / 4-10 dni / >10 dni).

Trening (z katalogu głównego projektu):
    python predict_stay.py train Dog
    python predict_stay.py train Cat

Predykcja dla nowego zwierzęcia (np. z agenta AI):
    from predict_stay import predict_stay
    predict_stay("Dog", {"sex_upon_intake": "Neutered Male", "age_upon_intake": "2 years",
                         "breed": "Labrador Retriever Mix", "color": "Black/White",
                         "intake_type": "Stray", "intake_condition": "Normal",
                         "name": None, "datetime": "2026-01-10 14:00:00"})
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, f1_score, log_loss, precision_recall_curve

from clean_data import feature_engineering_days

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "austin" / "raw"
MODELS = ROOT / "models"

BINS = [-np.inf, 4, 10, np.inf]  # klasa 0: <=4 dni, 1: 4-10 dni, 2: >10 dni
RECALL_TARGET = 0.8              # cel recall dla klasy >30 dni przy wyborze progu

# Kolumny, które nie są znane w chwili przyjęcia albo nie są cechami
DROP_COLS = [
    "target", "time_in_shelter_days", "time_in_shelter",
    "outcome_type", "outcome_subtype", "sex_upon_outcome",
    "intake_datetime", "animal_type", "young_and_cat", "datetime", "animal_id_intake"
]

# Pola, których wymaga feature_engineering_days (dla nowego zwierzęcia)
REQUIRED = [
    "animal_type", "sex_upon_intake", "age_upon_intake", "breed", "color",
    "intake_type", "intake_condition", "found_location", "name", "datetime",
]


def make_X(df):
    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns]).copy()

    bool_cols = X.select_dtypes(include=["bool"]).columns
    X[bool_cols] = X[bool_cols].astype(int)

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    X[cat_cols] = X[cat_cols].fillna("Unknown").astype(str)

    return X, cat_cols


def load_training_data(animal):
    intakes = pd.read_csv(RAW / "aac_intakes.csv")
    outcomes = pd.read_csv(RAW / "aac_intakes_outcomes.csv")

    intakes["datetime"] = pd.to_datetime(intakes["datetime"])
    outcomes["intake_datetime"] = pd.to_datetime(outcomes["intake_datetime"])

    df = intakes.merge(
        outcomes[["animal_id_intake", "intake_datetime", "time_in_shelter_days"]],
        left_on=["animal_id", "datetime"],
        right_on=["animal_id_intake", "intake_datetime"],
        how="inner",
    )

    df = df[df["animal_type"] == animal].copy()
    df = feature_engineering_days(df)
    df = df.sort_values("intake_datetime").reset_index(drop=True)
    df = df.dropna(subset=["time_in_shelter_days"]).reset_index(drop=True)
    df["target"] = pd.cut(df["time_in_shelter_days"], bins=BINS, labels=[0, 1, 2]).astype(int)
    
    return df


def train(animal):
    df = load_training_data(animal)
    X, cat_cols = make_X(df)
    y = df["target"].to_numpy()

    # Podział czasowy 70/15/15 (dane posortowane po dacie przyjęcia)
    n = len(X)
    i1, i2 = int(n * 0.70), int(n * 0.85)
    X_tr, X_va, X_te = X.iloc[:i1], X.iloc[i1:i2], X.iloc[i2:]
    y_tr, y_va, y_te = y[:i1], y[i1:i2], y[i2:]

    model = CatBoostClassifier(
        iterations=2000, learning_rate=0.03, depth=6,
        loss_function="MultiClass", random_seed=42, verbose=200,
        early_stopping_rounds=100,
    )
    model.fit(X_tr, y_tr, cat_features=cat_cols, eval_set=(X_va, y_va), use_best_model=True)

    # Próg dla klasy >10 dni wybrany na walidacji
    p_va_long = model.predict_proba(X_va)[:, 2]
    prec, rec, thr = precision_recall_curve((y_va == 2).astype(int), p_va_long)
    ok = rec[:-1] >= RECALL_TARGET
    threshold_long = float(thr[ok][np.argmax(prec[:-1][ok])]) if ok.any() else 0.5

    # Ocena na teście (tylko raz, na końcu)
    p_te = model.predict_proba(X_te)
    y_te_long = (y_te == 2).astype(int)
    flag = p_te[:, 2] >= threshold_long
    tp = np.sum(flag & (y_te_long == 1))

    print(f"\n[{animal}] train={len(X_tr)} val={len(X_va)} test={len(X_te)}")
    print(f"Log loss (test):          {log_loss(y_te, p_te, labels=[0, 1, 2]):.4f}")
    print(f"Macro F1 (test):          {f1_score(y_te, p_te.argmax(axis=1), average='macro'):.4f}")
    print(f"PR-AUC dla >10 dni:       {average_precision_score(y_te_long, p_te[:, 2]):.4f}")
    print(f"Próg dla >10 dni:         {threshold_long:.3f}")
    print(f"Oznaczone (flag rate):    {flag.mean():.1%}")
    print(f"Precyzja dla >10 dni:     {tp / max(flag.sum(), 1):.3f}")
    print(f"Recall dla >10 dni:       {tp / max(y_te_long.sum(), 1):.3f}")

    MODELS.mkdir(exist_ok=True)
    model.save_model(str(MODELS / f"{animal.lower()}_days.cbm"))
    meta = {
        "columns": X.columns.tolist(),
        "cat_features": cat_cols,
        "threshold_long": threshold_long,
    }
    (MODELS / f"{animal.lower()}_days.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Zapisano model i metadane w: {MODELS}")


_cache = {}


def _load(animal):
    key = animal.lower()
    if key not in _cache:
        model = CatBoostClassifier()
        model.load_model(str(MODELS / f"{key}_days.cbm"))
        meta = json.loads((MODELS / f"{key}_days.json").read_text(encoding="utf-8"))
        _cache[key] = (model, meta)
    return _cache[key]


def predict_stay(animal, record):
    """Zwraca prawdopodobieństwa czasu pobytu dla nowego zwierzęcia.

    animal: "Dog" albo "Cat"
    record: słownik z polami znanymi w chwili przyjęcia (brakujące pola można pominąć)
    """
    model, meta = _load(animal)

    row = {k: record.get(k) for k in REQUIRED}
    row["animal_type"] = animal
    row["datetime"] = pd.to_datetime(row["datetime"]) if row["datetime"] else pd.Timestamp.now()
    for k in ["sex_upon_intake", "age_upon_intake", "breed", "color",
              "intake_type", "intake_condition", "found_location"]:
        row[k] = row[k] or ""

    df = feature_engineering_days(pd.DataFrame([row]))
    X, _ = make_X(df)
    X = X.reindex(columns=meta["columns"])
    for c in meta["cat_features"]:
        X[c] = X[c].fillna("Unknown").astype(str)

    p = model.predict_proba(X)[0]
    idx = list(model.classes_)
    p_short, p_mid, p_long = p[idx.index(0)], p[idx.index(1)], p[idx.index(2)]

    return {
        "p_do_4_dni": round(float(p_short), 3),
        "p_4_10_dni": round(float(p_mid), 3),
        "p_powyzej_10_dni": round(float(p_long), 3),
        "flaga_dlugi_pobyt": bool(p_long >= meta["threshold_long"]),
    }


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "train":
        train(sys.argv[2])
    else:
        print(__doc__)
