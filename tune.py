import json
import optuna
import pandas as pd
import wandb
from dotenv import load_dotenv

from features import preprocess
from train import run_cv, RANDOM_SEED

load_dotenv()

STUDY_NAME = "catboost-v1"
N_TRIALS = 50


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


if __name__ == "__main__":
    df = pd.read_csv("data/train.csv")
    X, y = preprocess(df, is_train=True)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="maximize",
        study_name=STUDY_NAME,
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    study.optimize(lambda trial: objective(trial, X, y), n_trials=N_TRIALS, show_progress_bar=True)

    print(f"\nBest CV:     {study.best_value:.5f}")
    print(f"Best params: {study.best_params}")

    with open("best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=2)

    print("Saved best_params.json")
