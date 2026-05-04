import pandas as pd
import wandb
from catboost import CatBoostClassifier
from dotenv import load_dotenv

from features import preprocess

load_dotenv()

model = CatBoostClassifier()
model.load_model("model.cbm")

test_df = pd.read_csv("data/test.csv")
X_test, passenger_ids = preprocess(test_df, is_train=False)

preds = model.predict(X_test).astype(bool)

submission = pd.DataFrame({"PassengerId": passenger_ids, "Transported": preds})
submission.to_csv("submission.csv", index=False)

n_transported = preds.sum()
run = wandb.init(
    project="spaceship-titanic",
    job_type="predict",
    config=model.get_params(),
)
wandb.log({
    "n_rows": len(submission),
    "n_transported": int(n_transported),
    "transport_rate": round(n_transported / len(submission), 4),
})
wandb.finish()

print(f"Saved submission.csv ({len(submission)} rows)")
print(f"Transported: {n_transported} / {len(submission)} ({n_transported / len(submission):.1%})")
