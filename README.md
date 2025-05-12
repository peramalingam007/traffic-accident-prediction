AI-Driven Traffic Accident Severity Prediction

A simple Streamlit web app for a college project to predict traffic accident severity based on weather and environmental conditions using a RandomForestClassifier.

Features





Input weather conditions (temperature, humidity, etc.) to predict accident severity (1-4).



Visualize feature importance of the trained model.



Built with Streamlit for easy deployment.

Prerequisites





Python 3.8+



Google Colab (for development and testing)



Git



Streamlit Community Cloud account (for deployment)



ngrok (optional, for testing in Colab)

Setup





Clone the repository:

git clone https://github.com/your-username/traffic-accident-prediction.git
cd traffic-accident-prediction



Install dependencies:

pip install -r requirements.txt



Generate dataset (if accidents.csv is missing):

python generate_dataset.py



Generate model:

python train_model.py



Run the app locally:

streamlit run app.py

Testing in Google Colab





Upload app.py, train_model.py, generate_dataset.py, and (optionally) accidents.csv to Colab.



Install dependencies:

!pip install streamlit>=1.20.0 scikit-learn>=1.2.0 pandas>=1.5.0 numpy>=1.23.0 matplotlib>=3.6.0 seaborn>=0.12.0 joblib>=1.2.0 pyngrok



Generate dataset (if needed):

!python generate_dataset.py



Generate model:

!python train_model.py



Set up ngrok:

!wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
!unzip ngrok-stable-linux-amd64.zip
!./ngrok authtoken YOUR_AUTH_TOKEN



Run the app:

!streamlit run app.py &>/dev/null &
!curl -s http://localhost:4040/api/tunnels | python3 -c \
    'import sys, json; print("Public URL:", json.load(sys.stdin)["tunnels"][0]["public_url"])'

Deployment

Deploy to Streamlit Community Cloud:





Sign in with GitHub.



Create a new app, select this repository, and set app.py as the main file.



Deploy to get a public URL.

License

MIT License (see LICENSE file).
