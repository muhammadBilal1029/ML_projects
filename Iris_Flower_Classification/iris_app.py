import os
import streamlit as st
import numpy as np
import joblib

model_path = ("iris_predictor.pkl")
accuracy_path = ("model_accuracy.pkl")

try:
    model = joblib.load(model_path)
    accuracy = joblib.load(accuracy_path)
except FileNotFoundError:
    st.error("❌ Model or accuracy file not found. Please ensure they are present.")
    st.stop()
st.title("🌸 Iris Flower Species Predictor")
st.write("Enter the flower's measurements:")
st.info(f"Model Accuracy: {accuracy * 100:.2f}%")
# User input
sepal_length = st.slider('Sepal Length (cm)', 4.0, 8.0, 5.0)
sepal_width = st.slider('Sepal Width (cm)', 2.0, 4.5, 3.0)
petal_length = st.slider('Petal Length (cm)', 1.0, 7.0, 4.0)
petal_width = st.slider('Petal Width (cm)', 0.1, 2.5, 1.0)

# Predict button
if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)
    st.success(f"The predicted species is: **{prediction[0].capitalize()}**")
