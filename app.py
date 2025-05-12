import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import subprocess
import time

# Function to generate model files
def generate_model_files():
    if not os.path.exists('accidents.csv'):
        st.error("accidents.csv not found. Please run generate_dataset.py first.")
        return False
    if not os.path.exists('train_model.py'):
        st.error("train_model.py not found. Please ensure it is in the repository.")
        return False
    
    st.info("Generating model.pkl and feature_names.pkl...")
    try:
        # Run train_model.py and capture output
        result = subprocess.run(['python', 'train_model.py'], capture_output=True, text=True)
        if result.returncode != 0:
            st.error(f"Error running train_model.py:\n{result.stderr}")
            return False
        if not (os.path.exists('model.pkl') and os.path.exists('feature_names.pkl')):
            st.error("Model files were not generated. Check train_model.py output.")
            return False
        return True
    except Exception as e:
        st.error(f"Exception while running train_model.py: {str(e)}")
        return False

# Regenerate model files if missing
if not (os.path.exists('model.pkl') and os.path.exists('feature_names.pkl')):
    success = generate_model_files()
    if not success:
        st.stop()

# Load model and feature names
try:
    with open('model.pkl', 'rb') as f:
        model = joblib.load(f)
    with open('feature_names.pkl', 'rb') as f:
        feature_names = joblib.load(f)
except Exception as e:
    st.error(f"Error loading model files: {str(e)}")
    st.stop()

# App title
st.title("Traffic Accident Severity Prediction")
st.write("Enter weather conditions to predict accident severity.")

# User inputs
temperature = st.slider("Temperature (°F)", -20.0, 120.0, 70.0)
humidity = st.slider("Humidity (%)", 0.0, 100.0, 50.0)
pressure = st.slider("Pressure (in)", 20.0, 35.0, 29.92)
visibility = st.slider("Visibility (mi)", 0.0, 10.0, 5.0)
wind_speed = st.slider("Wind Speed (mph)", 0.0, 50.0, 10.0)
weather_conditions = ['Clear', 'Rain', 'Snow', 'Fog', 'Cloudy', 'Thunderstorm', 'Haze']
weather = st.selectbox("Weather Condition", weather_conditions)

# Create input DataFrame
input_data = pd.DataFrame(0, index=[0], columns=feature_names)
input_data['Temperature(F)'] = temperature
input_data['Humidity(%)'] = humidity
input_data['Pressure(in)'] = pressure
input_data['Visibility(mi)'] = visibility
input_data['Wind_Speed(mph)'] = wind_speed
if weather != 'Clear':
    weather_col = f"Weather_Condition_{weather}"
    if weather_col in input_data.columns:
        input_data[weather_col] = 1

# Predict
if st.button("Predict"):
    try:
        prediction = model.predict(input_data)[0]
        st.success(f"Predicted Severity: {prediction}")
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")

# Feature importance plot
st.subheader("Feature Importance")
fig, ax = plt.subplots(figsize=(8, 5))
feature_importance = pd.Series(model.feature_importances_, index=feature_names)
sns.barplot(x=feature_importance.values, y=feature_importance.index, ax=ax)
ax.set_xlabel("Importance")
st.pyplot(fig)
