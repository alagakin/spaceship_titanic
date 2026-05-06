import pandas as pd
import joblib
import wandb
from dotenv import load_dotenv

load_dotenv()

pipeline = joblib.load("pipeline.pkl")
test_df = pd.read_csv("data/test.csv")
passenger_ids = test_df['PassengerId'].copy()

preds = pipeline.predict(test_df).astype(bool)

submission = pd.DataFrame({"PassengerId": passenger_ids, "Transported": preds})
submission.to_csv("submission.csv", index=False)

n_transported = preds.sum()
wandb.init(
    project="spaceship-titanic",
    job_type="predict",
    config=pipeline.named_steps['model'].get_params(),
)
wandb.log({
    "n_rows": len(submission),
    "n_transported": int(n_transported),
    "transport_rate": round(n_transported / len(submission), 4),
})
wandb.finish()

print(f"Saved submission.csv ({len(submission)} rows)")
print(f"Transported: {n_transported} / {len(submission)} ({n_transported / len(submission):.1%})")
