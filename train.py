import json
import numpy as np
import pandas as pd
import wandb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
from dotenv import load_dotenv

from features import preprocess, CAT_FEATURES

load_dotenv()

RANDOM_SEED = 42
CV_FOLDS = 5


def run_cv(params, X, y):
    kf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    scores = []
    for train_idx, val_idx in kf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model = CatBoostClassifier(
            **params,
            cat_features=CAT_FEATURES,
            random_seed=RANDOM_SEED,
            verbose=0,
        )
        model.fit(X_train, y_train)
        scores.append(accuracy_score(y_val, model.predict(X_val)))
    return float(np.mean(scores)), float(np.std(scores))


def train_final(X, y, params):
    model = CatBoostClassifier(
        **params,
        cat_features=CAT_FEATURES,
        random_seed=RANDOM_SEED,
        verbose=100,
    )
    model.fit(X, y)
    return model


if __name__ == "__main__":
    df = pd.read_csv("data/train.csv")
    X, y = preprocess(df, is_train=True)

    with open("best_params.json") as f:
        params = json.load(f)

    print(f"Params: {params}\n")
    mean, std = run_cv(params, X, y)
    print(f"CV: {mean:.5f} ± {std:.5f}")

    print("\nTraining on full dataset...")
    model = train_final(X, y, params)
    model.save_model("model.cbm")
    print("Saved model.cbm")

    run = wandb.init(project="spaceship-titanic", job_type="train", config=params)
    wandb.log({"cv_mean": mean, "cv_std": std})
    wandb.finish()
