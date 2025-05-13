import gradio as gr
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import subprocess

# Function to generate model files
def generate_model_files():
    if not os.path.exists('accidents.csv'):
        raise FileNotFoundError("accidents.csv not found. Please run generate_dataset.py first.")
    if not os.path.exists('train_model.py'):
        raise FileNotFoundError("train_model.py not found. Please ensure it is in the repository.")
    print("Generating model.pkl and feature_names.pkl...")
    result = subprocess.run(['python', 'train_model.py'], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Error running train_model.py:\n{result.stderr}")
    if not (os.path.exists('model.pkl') and os.path.exists('feature_names.pkl')):
        raise RuntimeError("Model files were not generated. Check train_model.py output.")

# Regenerate model files if missing
if not (os.path.exists('model.pkl') and os.path.exists('feature_names.pkl')):
    generate_model_files()

# Load model and feature names
try:
    with open('model.pkl', 'rb') as f:
        model = joblib.load(f)
    with open('feature_names.pkl', 'rb') as f:
        feature_names = joblib.load(f)
except Exception as e:
    raise RuntimeError(f"Error loading model files: {str(e)}")

# Prediction function
def predict(temperature, humidity, pressure, visibility, wind_speed, weather):
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
    try:
        prediction = model.predict(input_data)[0]
        return f"Predicted Severity: {prediction}"
    except Exception as e:
        return f"Prediction error: {str(e)}"

# Feature importance plot
def plot_feature_importance():
    fig, ax = plt.subplots(figsize=(8, 5))
    feature_importance = pd.Series(model.feature_importances_, index=feature_names)
    sns.barplot(x=feature_importance.values, y=feature_importance.index, ax=ax)
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance")
    return fig

# Gradio interface
with gr.Blocks() as app:
    gr.Markdown("# Traffic Accident Severity Prediction")
    gr.Markdown("Enter weather conditions to predict accident severity.")
    temperature = gr.Slider(-20, 120, value=70, label="Temperature (°F)")
    humidity = gr.Slider(0, 100, value=50, label="Humidity (%)")
    pressure = gr.Slider(20, 35, value=29.92, label="Pressure (in)")
    visibility = gr.Slider(0, 10, value=5, label="Visibility (mi)")
    wind_speed = gr.Slider(0, 50, value=10, label="Wind Speed (mph)")
    weather = gr.Dropdown(['Clear', 'Rain', 'Snow', 'Fog', 'Cloudy', 'Thunderstorm', 'Haze'], label="Weather Condition")
    predict_button = gr.Button("Predict")
    output = gr.Textbox(label="Prediction")
    plot_button = gr.Button("Show Feature Importance")
    plot_output = gr.Plot(label="Feature Importance")
    predict_button.click(predict, inputs=[temperature, humidity, pressure, visibility, wind_speed, weather], outputs=output)
    plot_button.click(plot_feature_importance, outputs=plot_output)

# Launch app (for local testing; comment out for Hugging Face Spaces)
# app.launch()
