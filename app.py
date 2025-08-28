
import streamlit as st
import numpy as np
import joblib

# Load model
loaded_model = joblib.load("lgbm_model.pkl")

st.set_page_config(page_title="California House Price Predictor", layout="wide")

st.title("				🏡 California House Price Predictor App")

# Two-column layout
col1, col2 = st.columns([1, 2])

# --- Left Column: Input sliders ---
with col1:
    st.subheader("🔧 Input Features")
    f1 = st.slider("Median Income (10k$)", 0.5, 15.0, 3.5, step=0.1)
    f2 = st.slider("House Age (years)", 1, 52, 29, step=1)
    f3 = st.slider("No of Rooms", 1.0, 15.0, 5.0, step=1)
    f4 = st.slider("No of Bedrooms", 0.5, 5.0, 1.0, step=1)
    f5 = st.slider("Neighborhood Population", 0, 5000, 1000, step=50)
    f6 = st.slider("Average Occupancy (family size)", 1.0, 10.0, 3.0, step=1)
    f7 = st.slider("Latitude", 32.0, 42.0, 35.0, step=0.01)
    f8 = st.slider("Longitude", -125.0, -114.0, -119.0, step=0.01)

    features = np.array([[f1, f2, f3, f4, f5, f6, f7, f8]])

    if st.button("🔮 Predict"):
        prediction = loaded_model.predict(features)[0]
        st.session_state.pred = prediction

# --- Right Column: Prediction Results ---
with col2:
    st.subheader("📊 Prediction Results")

    if "pred" in st.session_state:
        prediction = st.session_state.pred
        st.success(f"🏠 Estimated Median House Value: **${prediction * 100000:,.2f}**")

        # Show house image
        st.image("/content/moiz1.jpg", width=300)

        # Fun animations
        st.balloons()
        st.progress(min(prediction / 5, 1.0))  # 5 is max MedHouseVal in dataset
    else:
        st.info("👉 Adjust the sliders and click **Predict** to see results here!")
