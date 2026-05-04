import argparse
import json
import numpy as np
import pandas as pd
import optuna
import wandb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
from dotenv import load_dotenv

from features import preprocess, CAT_FEATURES

load_dotenv()

RANDOM_SEED = 42
CV_FOLDS = 5
STUDY_NAME = "catboost-v1"
N_TRIALS = 50


def run_cv(params, X, y) -> float:
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


def objective(trial, X, y):
    params = {
        "iterations":          trial.suggest_int("iterations", 500, 5000),
        "learning_rate":       trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "depth":               trial.suggest_int("depth", 3, 10),
        "l2_leaf_reg":         trial.suggest_float("l2_leaf_reg", 1, 10, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1),
    }

    wandb.init(project="spaceship-titanic", group=STUDY_NAME, config=params, reinit=True,
               settings=wandb.Settings(silent=True))

    mean, std = run_cv(params, X, y)
    wandb.log({"cv_mean": mean, "cv_std": std})
    wandb.finish()

    return mean


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true", help="CV with params from best_params.json")
    args = parser.parse_args()

    df = pd.read_csv("data/train.csv")
    X, y = preprocess(df, is_train=True)

    if args.eval:
        with open("best_params.json") as f:
            params = json.load(f)
        print(f"Params: {params}\n")
        mean, std = run_cv(params, X, y)
        print(f"CV: {mean:.4f} ± {std:.4f}")

    else:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = optuna.create_study(
            direction="maximize",
            study_name=STUDY_NAME,
            sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        )
        study.optimize(lambda trial: objective(trial, X, y), n_trials=N_TRIALS, show_progress_bar=True)

        print(f"\nBest CV:     {study.best_value:.4f}")
        print(f"Best params: {study.best_params}")

        with open("best_params.json", "w") as f:
            json.dump(study.best_params, f, indent=2)

        print("\nRetraining on full dataset...")
        model = train_final(X, y, study.best_params)
        model.save_model("model.cbm")
        print("Done. Saved model.cbm and best_params.json")
