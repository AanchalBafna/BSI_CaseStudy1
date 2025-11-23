import pandas as pd
import numpy as np

np.random.seed(42)
stocks = [f"STK{i}" for i in range(1, 11)]
rows = []

for s in stocks:
    turnover = np.random.lognormal(mean=10, sigma=0.3, size=60)

    # Inject anomalies intentionally
    if s in ['STK4', 'STK8']:
        turnover[45:50] *= 4

    for i in range(60):
        rows.append([s, i+1, turnover[i]])

df = pd.DataFrame(rows, columns=["Stock", "Day", "Turnover"])
df.to_csv("Case1_turnover_outliers/data/turnover.csv", index=False)

print("Turnover data generated successfully! -> turnover.csv")
