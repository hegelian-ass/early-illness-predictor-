import streamlit as st
import numpy as np
import pickle

# Load model
model = pickle.load(open('model.pkl', 'rb'))

st.set_page_config(page_title="Cancer Predictor", layout="wide")

st.title("🧬 Breast Cancer Prediction App")
st.write("Enter tumor details to predict whether it is Benign or Malignant")

# -------- INPUTS -------- #

st.header("Mean Features")
col1, col2, col3 = st.columns(3)

with col1:
    radius_mean = st.number_input("Radius Mean", value=14.0)
    texture_mean = st.number_input("Texture Mean", value=20.0)
    perimeter_mean = st.number_input("Perimeter Mean", value=95.0)

with col2:
    area_mean = st.number_input("Area Mean", value=700.0)
    smoothness_mean = st.number_input("Smoothness Mean", value=0.10)
    compactness_mean = st.number_input("Compactness Mean", value=0.12)

with col3:
    concavity_mean = st.number_input("Concavity Mean", value=0.11)
    concave_points_mean = st.number_input("Concave Points Mean", value=0.06)
    symmetry_mean = st.number_input("Symmetry Mean", value=0.18)

st.header("Standard Error Features")
col4, col5 = st.columns(2)

with col4:
    radius_se = st.number_input("Radius SE", value=0.45)
    perimeter_se = st.number_input("Perimeter SE", value=3.5)
    area_se = st.number_input("Area SE", value=40.0)

with col5:
    compactness_se = st.number_input("Compactness SE", value=0.02)
    concavity_se = st.number_input("Concavity SE", value=0.03)
    concave_points_se = st.number_input("Concave Points SE", value=0.01)

st.header("Worst Features")
col6, col7, col8 = st.columns(3)

with col6:
    radius_worst = st.number_input("Radius Worst", value=16.2)
    texture_worst = st.number_input("Texture Worst", value=25.4)
    perimeter_worst = st.number_input("Perimeter Worst", value=110.5)

with col7:
    area_worst = st.number_input("Area Worst", value=850.0)
    smoothness_worst = st.number_input("Smoothness Worst", value=0.14)
    compactness_worst = st.number_input("Compactness Worst", value=0.30)

with col8:
    concavity_worst = st.number_input("Concavity Worst", value=0.35)
    concave_points_worst = st.number_input("Concave Points Worst", value=0.15)
    symmetry_worst = st.number_input("Symmetry Worst", value=0.28)

# -------- PREDICTION -------- #

if st.button("🔍 Predict"):

    input_data = np.array([[
        radius_mean, texture_mean, perimeter_mean, area_mean,
        smoothness_mean, compactness_mean, concavity_mean,
        concave_points_mean, symmetry_mean,
        radius_se, perimeter_se, area_se,
        compactness_se, concavity_se, concave_points_se,
        radius_worst, texture_worst, perimeter_worst,
        area_worst, smoothness_worst, compactness_worst,
        concavity_worst, concave_points_worst, symmetry_worst
    ]])

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    if prediction[0] == 1:
        st.error("⚠️ Malignant (Cancerous)")
    else:
        st.success("✅ Benign (Non-Cancerous)")

    st.write(f"Confidence: {np.max(probability)*100:.2f}%")