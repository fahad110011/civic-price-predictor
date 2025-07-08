import os
import joblib
import pandas as pd
import streamlit as st

# ───────────────────────────── Paths ──────────────────────────────
DATA_PATH  = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'civic_raw.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'pipe.pkl')

# ───────────────────────────── Helpers ────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

df    = load_data()
model = load_model()

# ───────────────────────────── UI ─────────────────────────────────
st.set_page_config(page_title="Honda Civic Price Predictor", layout="centered")
st.title("Honda Civic Price Predictor")
st.markdown("Enter your car’s specs and click **Predict** to estimate market price.")

# Sidebar inputs
st.sidebar.header("Car Features")
year        = st.sidebar.slider("Year", int(df["year"].min()), int(df["year"].max()), int(df["year"].median()))
odometer    = st.sidebar.slider("Odometer (miles)", int(df["odometer"].min()), int(df["odometer"].max()), int(df["odometer"].median()))
condition   = st.sidebar.selectbox("Condition", sorted(df["condition"].dropna().unique()))
transmission= st.sidebar.selectbox("Transmission", sorted(df["transmission"].dropna().unique()))
state       = st.sidebar.selectbox("State", sorted(df["state"].unique()))

# Predict button
if st.sidebar.button("Predict Price"):
    X_new = pd.DataFrame([{
        "year": year,
        "odometer": odometer,
        "condition": condition,
        "transmission": transmission,
        "state": state
    }])
    price = model.predict(X_new)[0]
    st.success(f"Estimated Price: **${price:,.0f}**")

# Optional raw data
if st.checkbox("Show raw data"):
    st.dataframe(df)
