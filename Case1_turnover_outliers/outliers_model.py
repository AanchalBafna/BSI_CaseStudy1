import os
import pandas as pd
from pyod.models.abod import ABOD
from pyod.models.knn import KNN
from pyod.models.iforest import IForest
from pyod.models.hbos import HBOS
from sklearn.preprocessing import MinMaxScaler


DATA_DIR = "Case1_turnover_outliers/data"
os.makedirs(DATA_DIR, exist_ok=True)


df = pd.read_csv(os.path.join(DATA_DIR, "turnover.csv"))

# Computed chg_turnover
df_avg = df.groupby("Stock")["Turnover"].apply(
    lambda x: x[30:].mean() - x[:30].mean()
).reset_index(name="chg_turnover")

# Fit the four detectors 
models = {
    "ABOD": ABOD(),
    "KNN": KNN(),
    "IForest": IForest(),
    "HBOS": HBOS()
}

score_cols = []
label_cols = []
for name, model in models.items():
    model.fit(df_avg[["chg_turnover"]])
    lbl_col = f"{name}_label"
    scr_col = f"{name}_score"
    df_avg[lbl_col] = model.labels_
    if hasattr(model, "decision_scores_"):
        df_avg[scr_col] = model.decision_scores_
    elif hasattr(model, "decision_function_"):
        df_avg[scr_col] = model.decision_function_(df_avg[["chg_turnover"]])
    else:
        df_avg[scr_col] = model.labels_

    score_cols.append(scr_col)
    label_cols.append(lbl_col)

mm = MinMaxScaler()
norm_cols = [c + "_norm" for c in score_cols]
df_avg[norm_cols] = mm.fit_transform(df_avg[score_cols])

df_avg["ensemble_score"] = df_avg[norm_cols].mean(axis=1)
df_avg["final_flag_vote"] = df_avg[label_cols].sum(axis=1).apply(lambda x: 1 if x >= 2 else 0)

threshold = df_avg["ensemble_score"].quantile(0.8)
df_avg["final_flag_score"] = (df_avg["ensemble_score"] >= threshold).astype(int)

df_avg["final_flag"] = df_avg["final_flag_vote"]

injected = ["STK4", "STK8"]
present = [s for s in injected if s in df_avg["Stock"].values]
if present:
    def eval_preds(col):
        preds = df_avg.set_index("Stock")[col]
        tp = int((preds.loc[present] == 1).sum())
        fn = int((preds.loc[present] == 0).sum())
        pred_pos = preds[preds == 1].index.tolist()
        prec = len([p for p in pred_pos if p in present]) / max(1, len(pred_pos))
        return {"tp": tp, "fn": fn, "precision": float(prec)}

    print("Injected anomalies present:", present)
    print("Vote eval:", eval_preds("final_flag_vote"))
    print("Score eval:", eval_preds("final_flag_score"))

# Save results
out_path = os.path.join(DATA_DIR, "outlier_results.csv")
df_avg.to_csv(out_path, index=False)
print(f"Outlier detection completed -> {out_path}")
