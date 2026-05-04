import pandas as pd
from catboost import CatBoostClassifier
from features import preprocess

model = CatBoostClassifier()
model.load_model("model.cbm")
test_df = pd.read_csv("data/test.csv")
X_test, passenger_ids = preprocess(test_df, is_train=False)

preds = model.predict(X_test).astype(bool)

submission = pd.DataFrame({"PassengerId": passenger_ids, "Transported": preds})
submission.to_csv("submission.csv", index=False)
print(f"Saved submission.csv ({len(submission)} rows)")
print(submission.head())
