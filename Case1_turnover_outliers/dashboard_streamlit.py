import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from datetime import datetime
import os
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False
try:
    from prophet import Prophet
except Exception:
    st.error("The `prophet` package is not installed in the Python environment used by Streamlit.\n\n" \
             "Install it using the Python executable shown in your environment, for example:\n\n" \
             "`C:\\Python313\\python.exe -m pip install prophet --use-pep517`\n\n" \
             "Then restart Streamlit.")
    st.stop()


st.set_page_config(page_title="BSE Market Surveillance AI Dashboard", layout="wide")

# ================== CUSTOM CSS ==================
st.markdown("""
<style>
body {background-color: #FFFFFF;}
h1,h2,h3,h4 {color:#071A40; font-family:'Montserrat'; font-weight:700;}
.kpi-card {
    background-color: #FFFFFF;
    border: 2px solid #0B4CCB;
    padding: 18px;
    border-radius: 14px;
    text-align: center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    margin-bottom: 10px;
}
.kpi-number {
    font-size: 38px;
    font-weight: 800;
    color: #0B4CCB;
    margin-top: -10px;
}
.kpi-label {
    font-size: 18px;
    color: #071A40;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<div style="background-color:#071A40; padding:20px; border-radius:12px; margin-bottom:18px;">
<h1 style="color:white; text-align:center;">BSE Market Surveillance & Risk Intelligence Dashboard</h1>
<p style="color:#AFC6FF; text-align:center;">Turnover Outliers • Risk Analytics • Forecasting</p>
</div>
""", unsafe_allow_html=True)


df = pd.read_csv("Case1_turnover_outliers/data/turnover.csv")
df_avg = pd.read_csv("Case1_turnover_outliers/data/outlier_results.csv")

try:
    df['Day'] = df['Day'].astype(int) - 1
    df['Date'] = pd.to_datetime(df['Day'], unit='D', origin='2020-01-01')
except Exception:
    df['Date'] = pd.to_datetime(df['Day'], errors='coerce')


df['Stock'] = df['Stock'].astype(str).str.strip()
df_avg['Stock'] = df_avg['Stock'].astype(str).str.strip()


st.sidebar.header("Filters & Options")
all_stocks = sorted(df['Stock'].unique())
default_sel = all_stocks[:10]

def _reset_filters():
    st.session_state['selected_stocks'] = default_sel

selected_stocks = st.sidebar.multiselect(
    "Select stocks (leave empty = all)",
    options=all_stocks,
    default=default_sel,
    key='selected_stocks'
)

st.sidebar.button("Reset filters", on_click=_reset_filters)

use_autodetect = st.sidebar.checkbox("Auto-detect high-risk stocks", value=True, help="Automatically pick stocks flagged as high risk by the outlier model")
top_n = st.sidebar.slider("Top N by change (for auto-selection)", min_value=1, max_value=max(1, len(df_avg)), value=3)


forecast_horizon = st.sidebar.slider("Forecast horizon (days)", min_value=1, max_value=60, value=15)


score_percentile = st.sidebar.slider("Ensemble flag percentile", min_value=50, max_value=99, value=80, help="Percentile cutoff on the ensemble score to mark stocks as high-risk by score")
st.sidebar.markdown("---")
st.sidebar.markdown("**Custom composition**")
composition_method = st.sidebar.selectbox("Composition method", options=["Voting (>=2)", "Ensemble percentile", "Custom thresholds"], index=0)


per_model_thresholds = {}



forecast_choice = st.sidebar.selectbox("Select stock for forecasting ", options=["Auto"] + all_stocks, index=0)


if not selected_stocks:
    selected_stocks = all_stocks

filtered_df_avg = df_avg[df_avg['Stock'].isin(selected_stocks)].reset_index(drop=True)


if 'ensemble_score' in df_avg.columns:
    score_thresh = df_avg['ensemble_score'].quantile(score_percentile/100.0)
    df_avg['final_flag_score'] = (df_avg['ensemble_score'] >= score_thresh).astype(int)
else:

    df_avg['final_flag_score'] = 0


filtered_df_avg = df_avg[df_avg['Stock'].isin(selected_stocks)].reset_index(drop=True)


high_risk_global = list(df_avg.loc[df_avg['final_flag'] == 1, 'Stock'])

high_risk_filtered = list(filtered_df_avg.loc[filtered_df_avg['final_flag'] == 1, 'Stock'])

high_risk_global_score = list(df_avg.loc[df_avg['final_flag_score'] == 1, 'Stock']) if 'final_flag_score' in df_avg.columns else []
high_risk_filtered_score = list(filtered_df_avg.loc[filtered_df_avg['final_flag_score'] == 1, 'Stock']) if 'final_flag_score' in filtered_df_avg.columns else []


norm_cols = [c for c in df_avg.columns if c.endswith('_norm')]
model_names = [c.replace('_score_norm','') for c in norm_cols]


if composition_method == 'Custom thresholds' and norm_cols:
    st.sidebar.markdown("Set per-model normalized-score thresholds (0-100)")
    for nc in norm_cols:
        label = nc.replace('_score_norm','')
        pct = st.sidebar.slider(f"{label} threshold %", min_value=0, max_value=100, value=50)
        per_model_thresholds[label] = pct/100.0


st.sidebar.markdown("---")
st.sidebar.markdown("**Hyperparameter sweep**")
sweep_button = st.sidebar.button("Run")
if sweep_button:
    st.sidebar.info("Running quick hyperparameter sweep — this runs models on `chg_turnover` and selects best params by injected-anomaly detection")
  
    knn_neighbors = [1,2,3,4,5]
    iforest_contams = [0.05, 0.1, 0.2]
    best = {"tp": -1, "precision": 0.0, "params": None}
   
    from pyod.models.knn import KNN as KNN_model
    from pyod.models.iforest import IForest as IForest_model
    from pyod.models.abod import ABOD as ABOD_model
    from pyod.models.hbos import HBOS as HBOS_model
    X = df_avg[["chg_turnover"]].fillna(0).values
    injected = ['STK4','STK8']
    for k in knn_neighbors:
        for cont in iforest_contams:
            knn = KNN_model(n_neighbors=k)
            ifor = IForest_model(contamination=cont)
            abod = ABOD_model()
            hbos = HBOS_model()
         
            try:
                abod.fit(X)
                knn.fit(X)
                ifor.fit(X)
                hbos.fit(X)
            except Exception as e:
                continue
           
            preds = pd.DataFrame({
                'ABOD': abod.labels_,
                'KNN': knn.labels_,
                'IForest': ifor.labels_,
                'HBOS': hbos.labels_
            }, index=df_avg['Stock'])
            vote = (preds.sum(axis=1) >= 2).astype(int)
            present = [s for s in injected if s in vote.index]
            if present:
                tp = int((vote.loc[present] == 1).sum())
                pred_pos = vote[vote==1].index.tolist()
                prec = len([p for p in pred_pos if p in present]) / max(1, len(pred_pos))
            else:
                tp = 0; prec = 0.0
            if (tp > best['tp']) or (tp == best['tp'] and prec > best['precision']):
                best = {'tp': tp, 'precision': prec, 'params': {'knn_n':k, 'iforest_contamination':cont}}
    st.sidebar.success(f"Best found: TP={best['tp']}, precision={best['precision']:.2f}, params={best['params']}")
    # save best params
    hp_path = os.path.join('Case1_turnover_outliers','data','hyperparam_results.json')
    with open(hp_path, 'w') as f:
        json.dump(best, f)
    st.sidebar.info(f"Saved hyperparameter sweep results -> {hp_path}")


n_top = int(max(1, min(top_n, len(df_avg))))

if use_autodetect:
  
    if high_risk_filtered:
        
        top_n_safe = max(1, min(top_n, max(1, len(high_risk_filtered))))
        auto_selected = list(high_risk_filtered[:top_n_safe])
    else:
        auto_selected = []
else:
    auto_selected = []

st.sidebar.markdown(f"**Auto-selected high-risk:** {', '.join(auto_selected) if auto_selected else 'None'}")

c1, c2, c3 = st.columns(3)
c1.markdown(f"<div class='kpi-card'><div class='kpi-label'>Total Stocks</div><div class='kpi-number'>{len(all_stocks)}</div></div>", unsafe_allow_html=True)

combined_flag_text = f"{len(high_risk_global)} total ({len(high_risk_filtered)} in selection)"
c2.markdown(f"<div class='kpi-card'><div class='kpi-label'>High Risk Stocks</div><div class='kpi-number'>{combined_flag_text}</div></div>", unsafe_allow_html=True)
c3.markdown(f"<div class='kpi-card'><div class='kpi-label'>Forecast Horizon</div><div class='kpi-number'>{forecast_horizon} Days</div></div>", unsafe_allow_html=True)



st.subheader("Outlier Detection Results")
st.dataframe(filtered_df_avg.reset_index(drop=True))


with st.expander("Debug: filter diagnostics", expanded=False):
    st.write("Selected stocks (sidebar):", st.session_state.get('selected_stocks'))
    st.write(f"Filtered rows in outlier results: {len(filtered_df_avg)}")
    st.write(filtered_df_avg.head(10))

# ================== VISUALIZATION ==================

st.subheader("📊 Overview & Evidence")
col_a, col_b = st.columns([2,3])

with col_a:
    st.markdown("**Change in Turnover (selected stocks)**")
    fig1, ax1 = plt.subplots(figsize=(6,3))
    plot_df = filtered_df_avg.copy()
    sns.barplot(data=plot_df, x="Stock", y="chg_turnover", palette="Blues_d", ax=ax1)
    ax1.set_ylabel("Change in Turnover")
    ax1.set_xlabel("")
    ax1.tick_params(axis='x', rotation=45)
    st.pyplot(fig1)

    st.markdown("**Distribution of Chg_turnover (selected)**")
    fig2, ax2 = plt.subplots(figsize=(6,2))
    sns.boxplot(data=plot_df["chg_turnover"], color="#0B4CCB", ax=ax2)
    st.pyplot(fig2)

with col_b:
    st.markdown("**Model Voting / Scores Heatmap (selected stocks)**")
    model_bases = ["ABOD", "KNN", "IForest", "HBOS"]
    available_cols = []
    col_display_names = []
    for m in model_bases:
        label_col = f"{m}_label"
        score_norm_col = f"{m}_score_norm"
        if label_col in df_avg.columns:
            available_cols.append(label_col)
            col_display_names.append(m)
        elif m in df_avg.columns:
            available_cols.append(m)
            col_display_names.append(m)
        elif score_norm_col in df_avg.columns:
            available_cols.append(score_norm_col)
            col_display_names.append(m + " (score)")
        else:
            continue

    if available_cols:
        heatmap_df = df_avg.set_index("Stock")[available_cols].reindex(selected_stocks).fillna(0)
        heatmap_df.columns = col_display_names
    else:
        heatmap_df = pd.DataFrame(0, index=selected_stocks, columns=["no_models_available"]).reset_index().set_index("index")

    fig3, ax3 = plt.subplots(figsize=(6,3))
    sns.heatmap(heatmap_df.fillna(0), cmap="Reds", annot=True, linewidths=0.5, ax=ax3)
    ax3.set_ylabel("")
    st.pyplot(fig3)


st.subheader("📈 Turnover Trend (Selected Stocks)")
trend_col1, trend_col2 = st.columns(2)
with trend_col1:
    fig4, ax4 = plt.subplots(figsize=(7,3))
    sample = df[df["Stock"].isin(selected_stocks)]
    sns.lineplot(data=sample, x="Date", y="Turnover", hue="Stock", linewidth=2, ax=ax4)
    ax4.set_title("Turnover Movement")
    ax4.tick_params(axis='x', rotation=45)
    st.pyplot(fig4)

with trend_col2:
    st.markdown("**High-risk Evidence**")
    st.write(f"Detected high-risk-filtered: {', '.join(high_risk_filtered) if high_risk_filtered else 'None'}")
    st.write(f"Detected high-risk-global: {', '.join(high_risk_global) if high_risk_global else 'None'}")
    top_movers = filtered_df_avg.nlargest(5, 'chg_turnover')
    st.table(top_movers[['Stock','chg_turnover','final_flag']].reset_index(drop=True))

    with st.expander("Model scores & Thresholding", expanded=False):
        st.write(f"Ensemble percentile cutoff: {score_percentile}% (threshold={score_thresh:.3f})")
        st.write(f"Detected high-risk (filtered, score-based): {', '.join(high_risk_filtered_score) if high_risk_filtered_score else 'None'}")
        st.write(f"Detected high-risk (global, score-based): {', '.join(high_risk_global_score) if high_risk_global_score else 'None'}")

        norm_cols = [c for c in df_avg.columns if c.endswith('_norm')]
        if norm_cols:
            nplots = len(norm_cols) + 1
            fig_scores, axes = plt.subplots(nplots, 1, figsize=(8, 2 * nplots), sharex=True, constrained_layout=True)
            if nplots == 1:
                axes = [axes]

            for i, col in enumerate(norm_cols):
                ax = axes[i]
                sns.histplot(df_avg[col], bins=8, ax=ax, color='#0B4CCB', edgecolor='k')
                ax.set_xlim(0, 1)
                ax.set_ylabel('Count')
                ax.set_title(col.replace('_score_norm', ''))
                ax.axvline(x=score_thresh, color='red', linestyle='--')

            ax = axes[-1]
            sns.histplot(df_avg['ensemble_score'], bins=12, ax=ax, color='#FF6B6B', edgecolor='k')
            ax.set_xlim(0, 1)
            ax.set_ylabel('Count')
            ax.set_title('Ensemble Score')
            ax.axvline(x=score_thresh, color='black', linestyle='--')

         
            for a in axes[:-1]:
                a.set_xlabel('')
            axes[-1].set_xlabel('Normalized score (0..1)')

            st.pyplot(fig_scores)
        else:
            st.write("No per-model normalized score columns found in `outlier_results.csv`.")

        
        injected = ['STK4','STK8']
        present = [s for s in injected if s in df_avg['Stock'].values]
        if present and 'final_flag_score' in df_avg.columns:
            preds = df_avg.set_index('Stock')['final_flag_score']
            tp = int((preds.loc[present] == 1).sum())
            fn = int((preds.loc[present] == 0).sum())
            pred_pos = preds[preds == 1].index.tolist()
            prec = len([p for p in pred_pos if p in present]) / max(1, len(pred_pos))
            st.markdown(f"**Injected anomalies present:** {present}")
            st.markdown(f"**Score-based detection:** TP={tp}, FN={fn}, precision={prec:.2f}")
        else:
            st.write("No injected anomalies present or score-based flags not available.")

    # ================== CUSTOM COMPOSITION & MONITORING ==================
    st.markdown("---")
    st.markdown("**Flag composition & monitoring**")
    if composition_method == 'Voting (>=2)':
        use_flag_col = 'final_flag'
    elif composition_method == 'Ensemble percentile':
        use_flag_col = 'final_flag_score'
    else:
        if norm_cols and per_model_thresholds:
            df_temp = df_avg.copy()
            for m in model_names:
                col = f"{m}_score_norm"
                thr = per_model_thresholds.get(m, 0.5)
                df_temp[f"{m}_custom_flag"] = (df_temp[col] >= thr).astype(int)
            combop = st.selectbox("Combine operator for custom flags", options=["OR", "AND", "Weighted"], index=0)
            if combop in ["OR","AND"]:
                if combop == 'OR':
                    df_temp['custom_final'] = (df_temp[[f"{m}_custom_flag" for m in model_names]].sum(axis=1) >= 1).astype(int)
                else:
                    df_temp['custom_final'] = (df_temp[[f"{m}_custom_flag" for m in model_names]].sum(axis=1) == len(model_names)).astype(int)
            else:
                weights = {}
                st.write("Set weights (sum to 1 recommended)")
                total = 0.0
                for m in model_names:
                    w = st.number_input(f"Weight for {m}", min_value=0.0, max_value=1.0, value=1.0/len(model_names))
                    weights[m] = w
                    total += w
                wsum = sum(weights.values()) or 1.0
                df_temp['weighted_score'] = 0.0
                for m in model_names:
                    df_temp['weighted_score'] += df_temp[f"{m}_score_norm"] * (weights[m]/wsum)
                wthr = st.slider('Weighted threshold (0-1)', 0.0, 1.0, 0.5)
                df_temp['custom_final'] = (df_temp['weighted_score'] >= wthr).astype(int)

            df_avg['final_flag_custom'] = df_temp['custom_final'].values
            use_flag_col = 'final_flag_custom'
        else:
            st.warning('No normalized model scores found — run outliers pipeline to generate `_norm` columns.')
            use_flag_col = 'final_flag'

    if st.button('Save flags snapshot'):
        flags_path = os.path.join('Case1_turnover_outliers','data','flags_history.csv')
        vote_list = df_avg.loc[df_avg['final_flag'] == 1, 'Stock'].tolist() if 'final_flag' in df_avg.columns else []
        score_list = df_avg.loc[df_avg['final_flag_score'] == 1, 'Stock'].tolist() if 'final_flag_score' in df_avg.columns else []
        custom_list = df_avg.loc[df_avg['final_flag_custom'] == 1, 'Stock'].tolist() if 'final_flag_custom' in df_avg.columns else []

        snapshot = pd.DataFrame({
            'timestamp': [datetime.utcnow().isoformat()],
            'flags_vote': [";".join(vote_list)],
            'flags_score': [";".join(score_list)],
            'flags_custom': [";".join(custom_list)],
            'ensemble_percentile': [score_percentile],
            'composition_method': [composition_method]
        })
        if not os.path.exists(flags_path):
            snapshot.to_csv(flags_path, index=False)
        else:
            snapshot.to_csv(flags_path, mode='a', header=False, index=False)
        st.success(f'Flags snapshot appended to {flags_path}')

# ================== FORECASTING SECTION ==================
st.subheader("📈 Forecasting")


if forecast_choice == "Auto":
    if auto_selected:
        forecast_stock = auto_selected[0]
    elif high_risk_filtered:
        forecast_stock = high_risk_filtered[0]
    else:
        forecast_stock = selected_stocks[0]
else:
    forecast_stock = forecast_choice

st.markdown(f"**Forecasting for:** {forecast_stock}")
stk_df = df[df["Stock"]==forecast_stock][["Date","Turnover"]].rename(columns={"Date":"ds","Turnover":"y"})

if stk_df['ds'].isna().any():
    st.error("Some date values could not be parsed for forecasting. Check the Day column in your CSV.")
else:
    horizon = forecast_horizon
    model = Prophet()
    with st.spinner(f"Training Prophet model for {forecast_stock}..."):
        model.fit(stk_df)
        future = model.make_future_dataframe(periods=horizon)
        forecast = model.predict(future)

    fig5 = model.plot(forecast)
    st.pyplot(fig5)

    fig6 = model.plot_components(forecast)
    st.pyplot(fig6)

    # ================== AUTO-GENERATED SUMMARY ==================
    def generate_summary(selected, high_risk, top_changes, observed, future_pred):
        lines = []
        lines.append(f"Selected {len(selected)} stocks for inspection.")
        if high_risk:
            lines.append(f"Auto-detected high-risk stocks: {', '.join(high_risk)}.")
        else:
            lines.append("No stocks currently flagged as high-risk by the model.")
        if not top_changes.empty:
            top = top_changes.iloc[0]
            lines.append(f"Top mover among selection: {top['Stock']} (change {top['chg_turnover']:.0f}).")

    
        if not observed.empty and not future_pred.empty:
            recent_mean = observed['y'].tail(7).mean()
            future_mean = future_pred['yhat'].tail(horizon).mean()
            pct = (future_mean - recent_mean) / recent_mean * 100 if recent_mean else 0
            trend = "increase" if pct > 0 else ("decrease" if pct < 0 else "no change")
            lines.append(f"Forecast: expected {trend} of {abs(pct):.1f}% vs recent (based on {horizon}-day horizon).")

        lines.append("Recommendation: monitor order books, trade clusters & corporate announcements for flagged stocks.")
        return "\n\n".join(lines)

    top_changes = filtered_df_avg.nlargest(3, 'chg_turnover')
    summary_text = generate_summary(selected_stocks, auto_selected or high_risk_filtered, top_changes, stk_df, forecast)
    st.subheader("Auto-generated Summary")
    st.write(summary_text)

    # ================== EXPLAINABILITY FOR SELECTED STOCK ==================
    with st.expander("Explainability (feature contributions)", expanded=False):
        feat_cols = [c for c in df_avg.columns if c not in ['Stock','final_flag','final_flag_score','final_flag_vote','ensemble_score'] and not c.endswith('_score') and not c.endswith('_norm')]
        prefer = ['chg_turnover','chg_turnover_pct','std_first30','std_last30','max_spike']
        cols = [c for c in prefer if c in df_avg.columns]
        if not cols:
            cols = [c for c in df_avg.columns if c.startswith(('chg','std','median','max'))]

        if cols and forecast_stock in df_avg['Stock'].values:
            row = df_avg.set_index('Stock').loc[forecast_stock]
            st.write(f"Feature snapshot for {forecast_stock}")
            z = ((df_avg[cols] - df_avg[cols].mean()) / (df_avg[cols].std() + 1e-9))
            z_stock = z.set_index(df_avg['Stock']).loc[forecast_stock]
            z_df = pd.DataFrame({'feature': cols, 'z_score': z_stock.values})
            z_df['abs_z'] = z_df['z_score'].abs()
            z_df = z_df.sort_values('abs_z', ascending=False)
            st.table(z_df.reset_index(drop=True))

            if HAS_SHAP:
                st.info('SHAP is available'
                try:
                    from pyod.models.iforest import IForest as IForest_model
                    X = df_avg[cols].fillna(0).values
                    ifor = IForest_model()
                    ifor.fit(X)
                    expl = shap.Explainer(ifor.predict, X)
                    shap_values = expl(X)
                    idx = df_avg.index[df_avg['Stock']==forecast_stock][0]
                    st.write('SHAP values for selected stock (approx):')
                    sv = pd.DataFrame({ 'feature': cols, 'shap': shap_values.values[idx] })
                    sv['abs_shap'] = sv['shap'].abs()
                    st.table(sv.sort_values('abs_shap', ascending=False).reset_index(drop=True))
                except Exception as e:
                    st.write('SHAP explanation failed:', e)
        else:
            st.write('No engineered numeric features found for explainability or selected stock missing from outlier results.')
