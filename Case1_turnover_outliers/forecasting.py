import pandas as pd
from prophet import Prophet

df = pd.read_csv("Case1_turnover_outliers/data/turnover.csv")

# taking example for STK4
stk = df[df["Stock"] == "STK4"][["Day","Turnover"]]
stk.columns = ["ds","y"]

.
try:
	stk['ds'] = stk['ds'].astype(int) - 1
	stk['ds'] = pd.to_datetime(stk['ds'], unit='D', origin='2020-01-01')
except Exception:
	stk['ds'] = pd.to_datetime(stk['ds'], errors='coerce')
	if stk['ds'].isna().any():
		bad = stk.loc[stk['ds'].isna(), 'ds'].unique()[:5]
		raise ValueError(f"Unable to parse ds values. Examples: {bad}")

model = Prophet()
model.fit(stk)

future = model.make_future_dataframe(periods=15)
forecast = model.predict(future)

forecast.to_csv("Case1_turnover_outliers/data/forecast_STK4.csv", index=False)
print("Forecast generated -> forecast_STK4.csv")
