import os
import joblib
import pandas as pd
import streamlit as st

DATA_PATH  = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'civic_raw.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'pipe.pkl')

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

df    = load_data()
model = load_model()

st.title("Honda Civic Price Predictor")
year        = st.slider("Year", 1990, 2025, 2015)
odometer    = st.slider("Odometer (mi)", 0, 300_000, 75_000, step=1_000)
condition   = st.selectbox("Condition", sorted(df["condition"].dropna().unique()))
trans       = st.selectbox("Transmission", sorted(df["transmission"].dropna().unique()))
state       = st.selectbox("State", sorted(df["state"].unique()))

if st.button("Predict Price"):
    X = pd.DataFrame([{
        "year": year,
        "odometer": odometer,
        "condition": condition,
        "transmission": trans,
        "state": state
    }])
    price = model.predict(X)[0]
    st.success(f"Estimated price: **${price:,.0f}**")
