import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="🏠 House Price Prediction", layout="centered")

st.title("🏠 House Price Prediction")
st.markdown("Enter the house details below to get an estimated price.")

# ── Load & train model ──────────────────────────────────────────────
@st.cache_resource
def load_model():
    house = pd.read_csv("Housing.csv")
    categorical_features = ['mainroad', 'guestroom', 'basement',
                             'hotwaterheating', 'airconditioning',
                             'prefarea', 'furnishingstatus']
    house_encoded = pd.get_dummies(house, columns=categorical_features, drop_first=True)
    X = house_encoded.drop(['price'], axis=1)
    y = house_encoded['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=100, max_depth=10,
                               min_samples_split=10, random_state=42)
    rf.fit(X_train, y_train)
    return rf, X_train.columns.tolist()

model, feature_cols = load_model()

# ── Input form ───────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    area        = st.number_input("Area (sq ft)", min_value=500, max_value=20000, value=2500, step=100)
    bedrooms    = st.selectbox("Bedrooms", [1, 2, 3, 4, 5, 6], index=2)
    bathrooms   = st.selectbox("Bathrooms", [1, 2, 3, 4], index=1)
    stories     = st.selectbox("Stories", [1, 2, 3, 4], index=1)
    parking     = st.selectbox("Parking Slots", [0, 1, 2, 3], index=1)

with col2:
    mainroad         = st.selectbox("Main Road Access", ["yes", "no"])
    guestroom        = st.selectbox("Guest Room", ["no", "yes"])
    basement         = st.selectbox("Basement", ["no", "yes"])
    hotwaterheating  = st.selectbox("Hot Water Heating", ["no", "yes"])
    airconditioning  = st.selectbox("Air Conditioning", ["no", "yes"])
    prefarea         = st.selectbox("Preferred Area", ["no", "yes"])
    furnishing       = st.selectbox("Furnishing Status", ["furnished", "semi-furnished", "unfurnished"])

# ── Predict ──────────────────────────────────────────────────────────
if st.button("🔍 Predict Price", use_container_width=True):
    input_data = {
        'area': area,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'stories': stories,
        'parking': parking,
        'mainroad_yes':              1 if mainroad == "yes" else 0,
        'guestroom_yes':             1 if guestroom == "yes" else 0,
        'basement_yes':              1 if basement == "yes" else 0,
        'hotwaterheating_yes':       1 if hotwaterheating == "yes" else 0,
        'airconditioning_yes':       1 if airconditioning == "yes" else 0,
        'prefarea_yes':              1 if prefarea == "yes" else 0,
        'furnishingstatus_semi-furnished': 1 if furnishing == "semi-furnished" else 0,
        'furnishingstatus_unfurnished':    1 if furnishing == "unfurnished" else 0,
    }
    df_input = pd.DataFrame([input_data])
    df_input = df_input.reindex(columns=feature_cols, fill_value=0)

    predicted = model.predict(df_input)[0]

    st.success(f"### 💰 Predicted House Price: ₹{predicted:,.0f}")
    st.caption("Prediction based on Random Forest model trained on the Housing dataset.")
