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
st.write("Enter road and weather conditions to predict accident severity.")

# User inputs
temperature = st.slider("Temperature (°F)", -20.0, 120.0, 70.0)
humidity = st.slider("Humidity (%)", 0.0, 100.0, 50.0)
pressure = st.slider("Pressure (in)", 20.0, 35.0, 29.92)
visibility = st.slider("Visibility (mi)", 0.0, 10.0, 5.0)
wind_speed = st.slider("Wind Speed (mph)", 0.0, 50.0, 10.0)

# Weather condition dropdown (example values; adjust based on your dataset)
weather_conditions = ['Clear', 'Rain', 'Snow', 'Fog', 'Cloudy']  # Adjust based on your dataset
weather = st.selectbox("Weather Condition", weather_conditions)

# Create input DataFrame
input_data = pd.DataFrame(0, index=[0], columns=feature_names)
input_data['Temperature(F)'] = temperature
input_data['Humidity(%)'] = humidity
input_data['Pressure(in)'] = pressure
input_data['Visibility(mi)'] = visibility
input_data['Wind_Speed(mph)'] = wind_speed

# Set the selected weather condition dummy variable to 1
if weather != 'Clear':  # Assuming 'Clear' is the reference category (dropped in get_dummies)
    weather_col = f"Weather_Condition_{weather}"
    if weather_col in input_data.columns:
        input_data[weather_col] = 1

# Predict
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    st.write(f"Predicted Accident Severity: **{prediction}**")

# Feature importance plot
st.subheader("Feature Importance")
fig, ax = plt.subplots()
feature_importance = pd.Series(model.feature_importances_, index=feature_names)
sns.barplot(x=feature_importance.values, y=feature_importance.index, ax=ax)
ax.set_xlabel("Importance")
st.pyplot(fig)
