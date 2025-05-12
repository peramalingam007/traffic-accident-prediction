import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and feature names
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('feature_names.pkl', 'rb') as f:
   
    feature_names = pickle.load(f)

# App title
st.title("AI-Driven Traffic Accident Severity Prediction")
st.write("Enter weather and environmental conditions to predict accident severity.")

# User inputs
temperature = st.slider("Temperature (°F)", -20.0, 120.0, 70.0, help="Temperature in Fahrenheit")
humidity = st.slider("Humidity (%)", 0.0, 100.0, 50.0, help="Relative humidity percentage")
pressure = st.slider("Pressure (in)", 20.0, 35.0, 29.92, help="Atmospheric pressure in inches")
visibility = st.slider("Visibility (mi)", 0.0, 10.0, 5.0, help="Visibility distance in miles")
wind_speed = st.slider("Wind Speed (mph)", 0.0, 50.0, 10.0, help="Wind speed in miles per hour")

# Weather condition dropdown (adjust based on dataset)
weather_conditions = ['Clear', 'Rain', 'Snow', 'Fog', 'Cloudy', 'Thunderstorm', 'Haze']
weather = st.selectbox("Weather Condition", weather_conditions, help="Select the current weather condition")

# Create input DataFrame
input_data = pd.DataFrame(0, index=[0], columns=feature_names)
input_data['Temperature(F)'] = temperature
input_data['Humidity(%)'] = humidity
input_data['Pressure(in)'] = pressure
input_data['Visibility(mi)'] = visibility
input_data['Wind_Speed(mph)'] = wind_speed

# Set the selected weather condition dummy variable to 1
if weather != 'Clear':  # Assuming 'Clear' is the reference category
    weather_col = f"Weather_Condition_{weather}"
    if weather_col in input_data.columns:
        input_data[weather_col] = 1

# Predict
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Accident Severity: **{prediction}**")

# Feature importance plot
st.subheader("Feature Importance")
fig, ax = plt.subplots(figsize=(10, 6))
feature_importance = pd.Series(model.feature_importances_, index=feature_names)
sns.barplot(x=feature_importance.values, y=feature_importance.index, ax=ax)
ax.set_xlabel("Importance")
ax.set_title("Feature Importance in RandomForestClassifier")
st.pyplot(fig)
