# app.py
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, auc
)
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

st.set_page_config(page_title="Insurance Risk & Retention Intelligence (v3)", layout="wide")

# ----------------- Helpers -----------------
@st.cache_data(show_spinner=False)
def load_csv(path_or_buffer):
    try:
        return pd.read_csv(path_or_buffer)
    except Exception:
        try:
            return pd.read_csv(path_or_buffer, sep=';')
        except Exception:
            return None

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    return df.dropna(axis=1, how='all').drop_duplicates()

def safe_numeric(series: pd.Series):
    return pd.to_numeric(series, errors='coerce')

def safe_bounds(series: pd.Series):
    s = safe_numeric(series)
    s = s[np.isfinite(s)]
    if s.empty:
        return None, None
    return float(s.min()), float(s.max())

def normalize_binary(y: pd.Series) -> pd.Series:
    if y is None:
        return None
    if y.dtype == object:
        lower = y.astype(str).str.lower()
        pos = {'yes','y','true','1','churn','claimed','madeclaim','claim','positive'}
        neg = {'no','n','false','0','renewed','retained','negative'}
        mapped = lower.map(lambda s: 1 if s in pos else (0 if s in neg else np.nan))
        if mapped.notna().mean() >= 0.5:
            # fill remaining by forward/backward to avoid NaNs in models
            return mapped.astype('float').fillna(method='ffill').fillna(method='bfill').fillna(0).astype(int)
        codes = pd.Categorical(y).codes
        codes = np.where(codes==-1, 0, codes)
        return pd.Series(codes, index=y.index)
    else:
        y2 = pd.to_numeric(y, errors='coerce')
        uniq = pd.Series(y2).dropna().unique()
        if not set(uniq).issubset({0,1}):
            med = np.nanmedian(y2)
            y2 = (y2 > med).astype(int)
        return pd.Series(y2, index=y.index)

def guess_columns(df: pd.DataFrame):
    cols = df.columns.tolist()
    def has(name):
        return next((c for c in cols if name.lower() in c.lower()), None)
    label = next((c for c in cols if any(k in c.lower() for k in ['churn','attrition','renew','claimmade','madeclaim','target'])), None)
    region = has('region')
    policytype = has('policy_type') or has('policy type') or has('policy')
    satis = has('satisf') or has('score')
    age = has('age')
    smoker = has('smoker') or has('smoking')
    bmi = has('bmi')
    charges = has('charges') or has('claim amount') or has('claim') or has('billed') or has('amount')
    premium = has('premium')
    segment = has('segment')
    return dict(label=label, region=region, policytype=policytype, satis=satis, age=age, smoker=smoker, bmi=bmi, charges=charges, premium=premium, segment=segment)

def build_preprocessor(X: pd.DataFrame):
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    pre = ColumnTransformer([
        ('num', SimpleImputer(strategy='median'), num_cols),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), cat_cols)
    ])
    return pre, num_cols, cat_cols

def split_xy(df: pd.DataFrame, label_col: str):
    X = df.drop(columns=[label_col]).copy()
    y = normalize_binary(df[label_col])
    return X, y

def class_balance_ok(y: pd.Series, desired_splits=5):
    if y is None or y.empty:
        return False, 0, "No target values."
    counts = pd.Series(y).value_counts(dropna=True)
    if len(counts) < 2:
        return False, 0, "Target has only one class after filtering; need at least two."
    min_count = int(counts.min())
    if min_count < 2:
        return False, 0, "Each class must have at least 2 samples."
    n_splits = min(desired_splits, min_count)
    if n_splits < 2:
        return False, 0, "Not enough samples to perform cross-validation."
    return True, n_splits, ""

def train_models(X, y, n_splits=5, random_state=42):
    pre, _, _ = build_preprocessor(X)
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=random_state),
        "Gradient Boosting": GradientBoostingClassifier(random_state=random_state)
    }
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rows, rocs = [], {}
    for name, clf in models.items():
        pipe = Pipeline([('prep', pre), ('clf', clf)])
        y_true, y_pred, y_prob = [], [], []
        for tr, te in skf.split(X, y):
            Xtr, Xte = X.iloc[tr], X.iloc[te]
            ytr, yte = y.iloc[tr], y.iloc[te]
            pipe.fit(Xtr, ytr)
            yp = pipe.predict(Xte)
            if hasattr(pipe.named_steps['clf'], 'predict_proba'):
                ypp = pipe.predict_proba(Xte)[:,1]
            else:
                try:
                    dec = pipe.decision_function(Xte)
                    ypp = (dec - dec.min()) / (dec.max() - dec.min() + 1e-9)
                except Exception:
                    ypp = yp.astype(float)
            y_true.extend(yte); y_pred.extend(yp); y_prob.extend(ypp)
        y_true, y_pred, y_prob = np.array(y_true), np.array(y_pred), np.array(y_prob)
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        try:
            auc_val = roc_auc_score(y_true, y_prob)
        except Exception:
            fpr, tpr, _ = roc_curve(y_true, y_pred); auc_val = auc(fpr, tpr)
        rows.append(dict(Algorithm=name,
                         **{"Testing Accuracy": round(acc,4),
                            "Precision": round(prec,4),
                            "Recall": round(rec,4),
                            "F1": round(f1,4),
                            "AUC": round(auc_val,4)}))
        try:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            rocs[name] = (fpr, tpr)
        except Exception:
            rocs[name] = (np.array([0,1]), np.array([0,1]))
    return pd.DataFrame(rows).set_index("Algorithm"), rocs, pre, models

# ----------------- Data load -----------------
st.sidebar.header("Dataset")
st.sidebar.caption("Place Insurance.csv in repo root or upload here. Nulls are handled automatically.")
upl = st.sidebar.file_uploader("Upload Insurance CSV (optional)", type=['csv'])
df = load_csv(upl) if upl is not None else None
if df is None:
    for fname in ["Insurance.csv", "sample_insurance.csv"]:
        df = load_csv(fname)
        if df is not None and not df.empty:
            break

if df is None or df.empty:
    st.error("No data found. Upload Insurance.csv or include it in the repo root.")
    st.stop()

df = clean_data(df)

# ----------------- Column mapping -----------------
st.sidebar.header("Column Mapping")
g = guess_columns(df)
def sb_select(label, key_guess):
    options = [None] + df.columns.tolist()
    default_idx = 0
    if key_guess in df.columns:
        default_idx = df.columns.tolist().index(key_guess) + 1
    return st.sidebar.selectbox(label, options, index=default_idx)

label_col   = sb_select("Target (Churn/Renewal/ClaimMade)", g['label'])
region_col  = sb_select("Region", g['region'])
policy_col  = sb_select("Policy Type", g['policytype'])
satis_col   = sb_select("Satisfaction (numeric for slider)", g['satis'])
age_col     = sb_select("Age (numeric)", g['age'])
smoker_col  = sb_select("Smoker", g['smoker'])
bmi_col     = sb_select("BMI", g['bmi'])
charges_col = sb_select("Charges / Claim Amount", g['charges'])
premium_col = sb_select("Premium", g['premium'])
segment_col = sb_select("Customer Segment (optional)", g['segment'])

# ----------------- Filters -----------------
st.sidebar.header("Filters (apply to charts)")
if region_col:
    regions = sorted(df[region_col].dropna().astype(str).unique().tolist())
    sel_regions = st.sidebar.multiselect("Region(s)", regions, default=regions[:min(3, len(regions))])
else:
    sel_regions = []

if policy_col:
    pols = sorted(df[policy_col].dropna().astype(str).unique().tolist())
    sel_pols = st.sidebar.multiselect("Policy Type(s)", pols, default=pols[:min(3, len(pols))])
else:
    sel_pols = []

slider_col = satis_col if satis_col else charges_col
if slider_col:
    lo, hi = safe_bounds(df[slider_col])
    if lo is None or hi is None or not np.isfinite(lo) or not np.isfinite(hi):
        s_lo, s_hi = None, None
        st.sidebar.warning(f"Slider disabled: '{slider_col}' lacks numeric values.")
    else:
        s_lo, s_hi = st.sidebar.slider(f"{slider_col} range", min_value=float(lo), max_value=float(hi), value=(float(lo), float(hi)))
else:
    s_lo, s_hi = None, None

def apply_filters(df):
    out = df.copy()
    if region_col and sel_regions:
        out = out[out[region_col].astype(str).isin(sel_regions)]
    if policy_col and sel_pols:
        out = out[out[policy_col].astype(str).isin(sel_pols)]
    if slider_col and s_lo is not None and s_hi is not None:
        out[slider_col] = safe_numeric(out[slider_col])
        out = out[(out[slider_col] >= s_lo) & (out[slider_col] <= s_hi)]
    return out

filtered = apply_filters(df)

# ----------------- Tabs -----------------
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview & Insights", "📈 Deep-Dive Charts", "🧠 Model Training & Evaluation", "🪄 Predict & Download"])

with tab1:
    st.markdown("## Overview & Insights")
    st.metric("Records (after filters)", len(filtered))
    if label_col and label_col in filtered.columns:
        ybin = normalize_binary(filtered[label_col])
        if ybin is not None and len(ybin) > 0:
            rate = float(np.nanmean(ybin))
            st.metric("Positive rate (target=1)", f"{(rate*100 if rate==rate else 0):.2f}%")
        else:
            st.info("Target column has insufficient data to compute rate.")

    st.divider()
    st.subheader("1) Average Claim Amount by Region and Policy Type (clustered bar)")
    if region_col and policy_col and charges_col:
        tmp = filtered[[region_col, policy_col, charges_col]].copy()
        tmp[charges_col] = safe_numeric(tmp[charges_col])
        tmp = tmp.dropna(subset=[charges_col])
        if not tmp.empty:
            agg = tmp.groupby([region_col, policy_col], dropna=False)[charges_col].mean().reset_index(name='AvgCharges')
            if not agg.empty:
                chart = alt.Chart(agg).mark_bar().encode(
                    x=alt.X(f'{region_col}:N', title='Region'),
                    y=alt.Y('AvgCharges:Q', title='Avg Claim Amount'),
                    color=alt.Color(f'{policy_col}:N', title='Policy Type'),
                    tooltip=[region_col, policy_col, alt.Tooltip('AvgCharges:Q', format=',.2f')]
                )
                st.altair_chart(chart, use_container_width=True)
            else:
                st.info("No aggregated data to show for the selected filters.")
        else:
            st.info("No numeric claim amounts after filtering.")

with tab2:
    st.markdown("## Deep-Dive Charts")
    st.caption("Five complementary visualizations. All reflect the global filters.")

    st.subheader("2) Satisfaction vs. Claim Amount (regression trend)")
    if satis_col and charges_col and satis_col in filtered.columns and charges_col in filtered.columns:
        t2 = filtered[[satis_col, charges_col]].copy()
        t2[satis_col] = safe_numeric(t2[satis_col])
        t2[charges_col] = safe_numeric(t2[charges_col])
        t2 = t2.dropna()
        if len(t2) >= 2:
            chart = alt.Chart(t2).mark_circle(opacity=0.4).encode(
                x=alt.X(f'{satis_col}:Q', title='Satisfaction'),
                y=alt.Y(f'{charges_col}:Q', title='Claim Amount'),
                tooltip=[satis_col, charges_col]
            ).properties(height=300)
            trend = chart.transform_regression(satis_col, charges_col, method='linear').mark_line()
            st.altair_chart(chart + trend, use_container_width=True)
        else:
            st.info("Not enough numeric data for regression chart.")
    else:
        st.info("Select Satisfaction and Charges/Claim Amount columns.")

    st.subheader("3) Heatmap: Age group × Smoking Status → Average Premium")
    if age_col and smoker_col and premium_col and all(c in filtered.columns for c in [age_col, smoker_col, premium_col]):
        tmp = filtered[[age_col, smoker_col, premium_col]].copy()
        tmp[age_col] = safe_numeric(tmp[age_col])
        tmp[premium_col] = safe_numeric(tmp[premium_col])
        tmp = tmp.dropna(subset=[age_col, premium_col])
        if not tmp.empty and tmp[age_col].nunique() > 1:
            # ensure at least 2 bins worth of data
            try:
                tmp['AgeBin'] = pd.cut(tmp[age_col], bins=min(5, max(2, int(tmp[age_col].nunique()))))
                agg = tmp.groupby(['AgeBin', smoker_col], dropna=False)[premium_col].mean().reset_index(name='AvgPremium')
                if not agg.empty:
                    hm = alt.Chart(agg).mark_rect().encode(
                        x=alt.X('AgeBin:O', title='Age Bucket'),
                        y=alt.Y(f'{smoker_col}:N', title='Smoker'),
                        color=alt.Color('AvgPremium:Q', title='Avg Premium', scale=alt.Scale(scheme='blues')),
                        tooltip=['AgeBin', smoker_col, alt.Tooltip('AvgPremium:Q', format=',.2f')]
                    )
                    st.altair_chart(hm, use_container_width=True)
                else:
                    st.info("No aggregated premium data available for heatmap.")
            except Exception as e:
                st.info(f"Could not build age bins: {e}")
        else:
            st.info("Insufficient numeric Age/Premium data for heatmap.")
    else:
        st.info("Select Age, Smoker, and Premium columns.")

    st.subheader("4) Feature Importance (Random Forest)")
    if label_col and label_col in filtered.columns:
        use_df = filtered.dropna(subset=[label_col])
        if not use_df.empty:
            X, y = split_xy(use_df, label_col)
            pre, num_cols, cat_cols = build_preprocessor(X)
            rf = RandomForestClassifier(n_estimators=300, random_state=42)
            pipe = Pipeline([('prep', pre), ('clf', rf)])
            try:
                pipe.fit(X, y)
                cat_features = []
                if len(cat_cols):
                    ohe = pipe.named_steps['prep'].named_transformers_['cat'].named_steps['onehot']
                    cat_features = list(ohe.get_feature_names_out(cat_cols))
                feat_names = list(num_cols) + cat_features
                imps = getattr(pipe.named_steps['clf'], 'feature_importances_', None)
                if imps is not None and len(imps) == len(feat_names):
                    imp_df = pd.DataFrame({'Feature': feat_names, 'Importance': imps}).sort_values('Importance', ascending=False).head(20)
                    chart = alt.Chart(imp_df).mark_bar().encode(
                        x=alt.X('Importance:Q'),
                        y=alt.Y('Feature:N', sort='-x'),
                        tooltip=['Feature', alt.Tooltip('Importance:Q', format='.4f')]
                    )
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.info("Feature importances unavailable for the current data.")
            except Exception as e:
                st.warning(f"Could not compute feature importance: {e}")
        else:
            st.info("Not enough rows with target to compute feature importance.")
    else:
        st.info("Select a target column for feature importance.")

    st.subheader("5) Bubble: BMI × Age vs Total Charges")
    if bmi_col and age_col and charges_col and all(c in filtered.columns for c in [bmi_col, age_col, charges_col]):
        tmp = filtered[[bmi_col, age_col, charges_col]].copy()
        tmp[bmi_col] = safe_numeric(tmp[bmi_col])
        tmp[age_col] = safe_numeric(tmp[age_col])
        tmp[charges_col] = safe_numeric(tmp[charges_col])
        tmp = tmp.dropna()
        if len(tmp) >= 2:
            bubble = alt.Chart(tmp).mark_circle(opacity=0.35).encode(
                x=alt.X(f'{bmi_col}:Q', title='BMI'),
                y=alt.Y(f'{age_col}:Q', title='Age'),
                size=alt.Size(f'{charges_col}:Q', title='Total Charges', legend=None),
                tooltip=[bmi_col, age_col, charges_col]
            ).interactive()
            st.altair_chart(bubble, use_container_width=True)
        else:
            st.info("Insufficient numeric data for bubble chart.")
    else:
        st.info("Select BMI, Age, and Charges columns.")

with tab3:
    st.markdown("## Model Training & Evaluation")
    if not label_col or label_col not in filtered.columns:
        st.error("Select a binary target column (e.g., Churn/Renewal/ClaimMade).")
    else:
        use_df = filtered.dropna(subset=[label_col])
        if use_df.empty:
            st.error("No rows available after filters/NA removal for the selected target.")
        else:
            X, y = split_xy(use_df, label_col)
            ok, n_splits, msg = class_balance_ok(y, desired_splits=5)
            if not ok:
                st.error(f"Cross-validation unavailable: {msg}")
            else:
                st.write(f"Using **{len(X)}** records | Stratified **{n_splits}-fold** CV.")
                go = st.button("Run CV on Decision Tree / Random Forest / Gradient Boosting")
                if go:
                    with st.spinner("Training..."):
                        metrics_df, roc_curves, preprocessor, models = train_models(X, y, n_splits=n_splits)
                    st.subheader("Metrics Table")
                    st.dataframe(metrics_df)

                    st.subheader("ROC Curves")
                    fig = plt.figure(figsize=(6,4))
                    for name, (fpr, tpr) in roc_curves.items():
                        plt.plot(fpr, tpr, label=name)
                    plt.plot([0,1], [0,1], linestyle='--')
                    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("ROC Curves"); plt.legend()
                    st.pyplot(fig)

                    st.subheader("Confusion Matrices (holdout split)")
                    try:
                        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
                    except Exception:
                        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)
                    for name, clf in models.items():
                        pipe = Pipeline([('prep', preprocessor), ('clf', clf)])
                        pipe.fit(Xtr, ytr)
                        yhat = pipe.predict(Xte)
                        cm = confusion_matrix(yte, yhat, labels=[0,1])
                        cm_df = pd.DataFrame(cm, index=['Actual 0','Actual 1'], columns=['Pred 0','Pred 1'])
                        st.write(f"**{name}**"); st.dataframe(cm_df)

with tab4:
    st.markdown("## Predict on New Data & Download Results")
    st.write("Upload a CSV **without the target**. We'll fit on current filtered data and return predictions.")
    uploaded = st.file_uploader("Upload CSV for prediction", type=['csv'])
    model_choice = st.selectbox("Model for final fit", ["Random Forest", "Decision Tree", "Gradient Boosting"])
    predict_btn = st.button("Fit on Current Data & Predict on Uploaded File")

    if predict_btn:
        if uploaded is None:
            st.error("Please upload a CSV file to predict.")
        elif not label_col or label_col not in filtered.columns:
            st.error("Please select the target column in the sidebar first.")
        else:
            use_df = filtered.dropna(subset=[label_col])
            if use_df.empty:
                st.error("No rows available for training. Adjust filters or column mapping.")
            else:
                X_full, y_full = split_xy(use_df, label_col)
                pre, _, _ = build_preprocessor(X_full)
                if model_choice == "Decision Tree":
                    clf = DecisionTreeClassifier(random_state=42)
                elif model_choice == "Gradient Boosting":
                    clf = GradientBoostingClassifier(random_state=42)
                else:
                    clf = RandomForestClassifier(n_estimators=300, random_state=42)
                pipe = Pipeline([('prep', pre), ('clf', clf)])
                pipe.fit(X_full, y_full)

                new_df = load_csv(uploaded)
                if new_df is None or new_df.empty:
                    st.error("Uploaded CSV could not be read or is empty.")
                else:
                    drop_cols = [c for c in new_df.columns if any(k in c.lower() for k in ['churn','attrition','renew','claimmade','madeclaim','target'])]
                    newX = new_df.drop(columns=drop_cols) if drop_cols else new_df.copy()
                    try:
                        if hasattr(pipe.named_steps['clf'], 'predict_proba'):
                            probs = pipe.predict_proba(newX)[:,1]
                        else:
                            preds_raw = pipe.predict(newX)
                            probs = preds_raw.astype(float)
                        preds = pipe.predict(newX)
                        out = new_df.copy()
                        out['prediction'] = preds
                        out['probability'] = probs
                        st.success("Predictions generated.")
                        st.dataframe(out.head(50))
                        csv_bytes = out.to_csv(index=False).encode('utf-8')
                        st.download_button("Download predictions CSV", data=csv_bytes, file_name="insurance_predictions.csv", mime="text/csv")
                    except Exception as e:
                        st.error(f"Prediction failed. Make sure columns are compatible. Error: {e}")

st.caption("v3: Added robust checks for empty datasets, non-numeric sliders, insufficient data for bins, and class imbalance in CV. Nulls are ignored or imputed (median/most-frequent).")
