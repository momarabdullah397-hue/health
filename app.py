import streamlit as st
import numpy as np
import joblib

st.title("ML Prediction App")

try:
    model = joblib.load("heart_disease_model.pkl")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.success("Model loaded successfully!")

feature1 = st.number_input("Feature 1", value=0.0)
feature2 = st.number_input("Feature 2", value=0.0)
feature3 = st.number_input("Feature 3", value=0.0)

if st.button("Predict"):
    try:
        features = np.array([[feature1, feature2, feature3]])
        prediction = model.predict(features)
        st.success(f"Prediction: {prediction[0]}")
    except Exception as e:
        st.error(f"Prediction error: {e}")
