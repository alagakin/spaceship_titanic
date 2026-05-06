import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

from features import build_pipeline


def train_final(df: pd.DataFrame, params: dict, *, random_seed: int = 42) -> object:
    X = df.drop(columns=['Transported'])
    y = df['Transported'].astype(int)
    pipeline = build_pipeline(params, random_seed=random_seed, verbose=100)
    pipeline.fit(X, y)
    joblib.dump(pipeline, "pipeline.pkl")
    print("Saved pipeline.pkl")
    return pipeline


def run_cv(params: dict, df: pd.DataFrame, *, cv_folds: int = 5, random_seed: int = 42) -> tuple[float, float]:
    X = df.drop(columns=['Transported'])
    y = df['Transported'].astype(int)

    kf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
    scores = []

    for train_idx, val_idx in kf.split(X, y):
        pipe = build_pipeline(params, random_seed=random_seed)
        pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = pipe.predict(X.iloc[val_idx])
        scores.append(accuracy_score(y.iloc[val_idx], preds))

    return float(np.mean(scores)), float(np.std(scores))
