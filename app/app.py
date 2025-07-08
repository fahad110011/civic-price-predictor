# ─── app/app.py ────────────────────────────────────────────────────────────────
import os
import joblib
import pandas as pd
import streamlit as st

ROOT       = os.path.dirname(os.path.dirname(__file__))
DATA_PATH  = os.path.join(ROOT, "data",  "raw",   "civic_raw.csv")
MODEL_PATH = os.path.join(ROOT, "model", "pipe.pkl")

# ------------------------------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

df    = load_data()
pipe  = load_model()                     # this is an sklearn.Pipeline

# helper: return the XGBRegressor inside the pipeline (last step)
def xgb_estimator():
    return pipe.steps[-1][1]             # e.g. ('xgb', XGBRegressor)

# ------------------------------------------------------------------------------
st.set_page_config(page_title="Honda Civic Price Predictor", page_icon="🚗")
st.title("🚗 Honda Civic Price Predictor")

with st.sidebar:
    st.header("Car Features")
    year = st.slider("Year",
                     int(df.year.min()), int(df.year.max()),
                     int(df.year.median()))
    odometer = st.slider("Odometer (miles)",
                         int(df.odometer.min()), int(df.odometer.max()),
                         int(df.odometer.median()), step=1_000)   # cast to int
    condition   = st.selectbox("Condition",
                               sorted(df.condition.dropna().unique()))
    transmission = st.selectbox("Transmission",
                                sorted(df.transmission.dropna().unique()))
    state = st.selectbox("State", sorted(df.state.unique()))
    predict_btn = st.button("Predict Price", use_container_width=True)

if predict_btn:
    X = pd.DataFrame([{
        "year": year,
        "odometer": odometer,
        "condition": condition,
        "transmission": transmission,
        "state": state
    }])

    try:
        price = pipe.predict(X)[0]                        # normal path
    except AttributeError as e:
        # old pickled model built with ancient xgboost: inject missing attrs
        if "'gpu_id'" in str(e):
            est = xgb_estimator()
            setattr(est, "gpu_id", -1)                    # fake CPU id
            price = pipe.predict(X)[0]                    # retry
        else:
            raise                                          # bubble up other errors

    st.success(f"Estimated price: **${price:,.0f}**")
# ───────────────────────────────────────────────────────────────────────────────
