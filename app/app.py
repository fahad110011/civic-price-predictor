import pickle
import streamlit as st
import pandas as pd

# 1. Load model
@st.cache_resource
def load_pipeline():
    with open("../model/pipe.pkl", "rb") as f:
        return pickle.load(f)

pipe = load_pipeline()

# 2. Page config
st.set_page_config(page_title="Honda Civic Price Predictor", layout="centered")

st.title("Honda Civic Price Predictor")
st.write("Enter the details below, and click **Predict** to estimate the market price.")

# 3. User inputs
year       = st.number_input("Year", min_value=1990, max_value=2025, value=2015)
odometer   = st.slider("Mileage (miles)", min_value=0, max_value=300_000, value=50_000, step=1_000)
condition  = st.selectbox("Condition", pipe.named_steps["pre"].named_transformers_["cat"]
                          .categories_[0].tolist())  # replace index if needed
trans      = st.selectbox("Transmission", pipe.named_steps["pre"].named_transformers_["cat"]
                          .categories_[1].tolist())
state      = st.selectbox("State", pipe.named_steps["pre"].named_transformers_["cat"]
                          .categories_[2].tolist())

# 4. Predict button
if st.button("Predict"):
    X = pd.DataFrame([{
        "year": year,
        "odometer": odometer,
        "condition": condition,
        "transmission": trans,
        "state": state
    }])
    price = pipe.predict(X)[0]
    st.metric("Estimated Price (USD)", f"${price:,.0f}")
