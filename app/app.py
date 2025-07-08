import joblib            # ← MUST be joblib
import streamlit as st
import pandas as pd
import os


# 1. Load data
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'civic_raw.csv')
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)   # ← joblib.load

# 2. Load model
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'pipe.pkl')
@st.cache_resource
import joblib
def load_model():
    return joblib.load(MODEL_PATH)

# 3. Build UI
st.title("Honda Civic Price Predictor")
st.markdown("Enter your car’s specs and see an estimated market price!")

df = load_data()
model = load_model()

# Sidebar for inputs
st.sidebar.header("Car Features")
year = st.sidebar.slider("Year", int(df['year'].min()), int(df['year'].max()), int(df['year'].median()))
odometer = st.sidebar.slider("Odometer (miles)", int(df['odometer'].min()), int(df['odometer'].max()), int(df['odometer'].median()))
condition = st.sidebar.selectbox("Condition", sorted(df['condition'].dropna().unique()))
transmission = st.sidebar.selectbox("Transmission", sorted(df['transmission'].dropna().unique()))
state = st.sidebar.selectbox("State", sorted(df['state'].unique()))

# 4. Predict
if st.sidebar.button("Predict Price"):
    X_new = pd.DataFrame([{
        'year': year,
        'odometer': odometer,
        'condition': condition,
        'transmission': transmission,
        'state': state
    }])
    price_pred = model.predict(X_new)[0]
    st.success(f"Estimated Price: ${price_pred:,.0f}")

# 5. Optional: show raw data
if st.checkbox("Show raw data"):
    st.dataframe(df)
