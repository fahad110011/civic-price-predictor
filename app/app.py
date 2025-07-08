# ─── app/app.py ────────────────────────────────────────────────────────────────
import os
import joblib
import pandas as pd
import streamlit as st

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT       = os.path.dirname(os.path.dirname(__file__))
DATA_PATH  = os.path.join(ROOT, "data",  "raw",   "civic_raw.csv")
MODEL_PATH = os.path.join(ROOT, "model", "pipe.pkl")

# ── Load & preprocess ──────────────────────────────────────────────────────────
@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    # Drop top 1% of odometer outliers so our slider & model stay realistic
    odo_99 = df.odometer.quantile(0.99)
    return df[df.odometer <= odo_99].copy()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

df   = load_data()
pipe = load_model()  # sklearn Pipeline

# ── Helpers ────────────────────────────────────────────────────────────────────
def xgb_estimator():
    """Return the XGBRegressor inside our pipeline (last step)."""
    return pipe.steps[-1][1]

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Honda Civic Price Predictor", page_icon="🚗")
st.title("🚗 Honda Civic Price Predictor")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Car Features")

    # Year slider
    year = st.slider(
        "Year",
        int(df.year.min()), int(df.year.max()),
        int(df.year.median())
    )

    # Odometer slider clamped at the 99th percentile
    odo_99 = int(df.odometer.quantile(0.99))
    odo_med = int(df.odometer.median())
    odometer = st.slider(
        "Odometer (miles)",
        0, odo_99, odo_med,
        step=1_000
    )

    condition    = st.selectbox("Condition",    sorted(df.condition.dropna().unique()))
    transmission = st.selectbox("Transmission", sorted(df.transmission.dropna().unique()))
    state        = st.selectbox("State",        sorted(df.state.unique()))

    predict_btn = st.button("Predict Price", use_container_width=True)

# ── Prediction ─────────────────────────────────────────────────────────────────
if predict_btn:
    X = pd.DataFrame([{
        "year": year,
        "odometer": odometer,
        "condition": condition,
        "transmission": transmission,
        "state": state
    }])

    try:
        price = pipe.predict(X)[0]
    except AttributeError as e:
        # handle ancient XGBoost pickle missing `gpu_id`
        if "'gpu_id'" in str(e):
            est = xgb_estimator()
            setattr(est, "gpu_id", -1)    # force CPU predictor
            price = pipe.predict(X)[0]
        else:
            raise

    st.success(f"Estimated price: **${price:,.0f}**")
