import json
import numpy as np
import pandas as pd
import joblib
import wandb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from dotenv import load_dotenv

from features import build_pipeline

load_dotenv()

RANDOM_SEED = 42
CV_FOLDS = 5


def train_final(df: pd.DataFrame, params: dict) -> object:
    X = df.drop(columns=['Transported'])
    y = df['Transported'].astype(int)
    pipeline = build_pipeline(params, random_seed=RANDOM_SEED, verbose=100)
    pipeline.fit(X, y)
    joblib.dump(pipeline, "pipeline.pkl")
    print("Saved pipeline.pkl")
    return pipeline


def run_cv(params: dict, df: pd.DataFrame, cv_folds: int = CV_FOLDS) -> tuple[float, float]:
    X = df.drop(columns=['Transported'])
    y = df['Transported'].astype(int)

    kf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)
    scores = []

    for train_idx, val_idx in kf.split(X, y):
        pipe = build_pipeline(params, random_seed=RANDOM_SEED)
        pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = pipe.predict(X.iloc[val_idx])
        scores.append(accuracy_score(y.iloc[val_idx], preds))

    return float(np.mean(scores)), float(np.std(scores))


if __name__ == "__main__":
    df = pd.read_csv("data/train.csv")

    with open("best_params.json") as f:
        params = json.load(f)

    print(f"Params: {params}\n")
    mean, std = run_cv(params, df)
    print(f"CV: {mean:.5f} ± {std:.5f}")

    print("\nTraining on full dataset...")
    train_final(df, params)

    from features import PIPELINE_STEPS
    wandb.init(project="spaceship-titanic", job_type="train", config={
        **params,
        "model": "catboost",
        "cv_folds": CV_FOLDS,
        "pipeline_steps": PIPELINE_STEPS,
    })
    wandb.log({"cv_mean": mean, "cv_std": std})
    wandb.finish()
