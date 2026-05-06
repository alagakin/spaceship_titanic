import json
import optuna
import pandas as pd
import wandb
from dotenv import load_dotenv

from train import run_cv, RANDOM_SEED, CV_FOLDS
from features import PIPELINE_STEPS

load_dotenv()

STUDY_NAME = "catboost-v1"
N_TRIALS = 2


def objective(trial, df, study_name, cv_folds):
    params = {
        "iterations":          trial.suggest_int("iterations", 500, 5000),
        "learning_rate":       trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "depth":               trial.suggest_int("depth", 3, 10),
        "l2_leaf_reg":         trial.suggest_float("l2_leaf_reg", 1, 10, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1),
    }

    wandb.init(project="spaceship-titanic", group=study_name, reinit=True,
               settings=wandb.Settings(silent=True), config={
                   **params,
                   "model": "catboost",
                   "cv_folds": cv_folds,
                   "study_name": study_name,
                   "pipeline_steps": PIPELINE_STEPS,
               })

    mean, std = run_cv(params, df, cv_folds=cv_folds)
    wandb.log({"cv_mean": mean, "cv_std": std})
    wandb.finish()

    return mean


def run_study(df: pd.DataFrame, n_trials: int = N_TRIALS, study_name: str = STUDY_NAME,
              cv_folds: int = CV_FOLDS):
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    study.optimize(
        lambda trial: objective(trial, df, study_name, cv_folds),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    print(f"\nBest CV:     {study.best_value:.5f}")
    print(f"Best params: {study.best_params}")

    with open("best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=2)

    print("Saved best_params.json")
    return study


if __name__ == "__main__":
    df = pd.read_csv("data/train.csv")
    run_study(df)
