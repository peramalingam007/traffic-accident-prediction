AI-Driven Traffic Accident Severity Prediction
This project uses a RandomForestClassifier to predict traffic accident severity based on weather and environmental conditions. The model is deployed as a Streamlit web app, allowing users to input features like temperature, humidity, and weather condition to predict accident severity.
Repository Structure
graph TD
    A[traffic-accident-prediction] --> B[.gitignore]
    A --> C[README.md]
    A --> D[requirements.txt]
    A --> E[app.py]
    A --> F[train_model.py]
    A --> G[model.pkl]
    A --> H[feature_names.pkl]
    A --> I[LICENSE]

Workflow
graph TD
    A[Develop in Google Colab] --> B[Run train_model.py]
    B --> C[Generate model.pkl & feature_names.pkl]
    C --> D[Test app.py with Streamlit]
    D --> E[Push Files to GitHub]
    E --> F[Deploy to Streamlit Community Cloud]
    F --> G[Update Code]
    G --> E

Features

Predict accident severity using inputs: Temperature(F), Humidity(%), Pressure(in), Visibility(mi), Wind_Speed(mph), Weather_Condition.
Visualize feature importance of the trained model.
Built with RandomForestClassifier and deployed via Streamlit.

Prerequisites

Python 3.8+
Google Colab (for training and testing)
Git
Streamlit Community Cloud account (for deployment)
ngrok (optional, for testing in Colab)

Setup

Clone the repository:git clone https://github.com/your-username/traffic-accident-prediction.git
cd traffic-accident-prediction


Install dependencies:pip install -r requirements.txt


Train the model (if not using pre-trained model.pkl):
Upload accidents.csv to Google Colab.
Run train_model.py to generate model.pkl and feature_names.pkl.


Run the app locally:streamlit run app.py



Dataset
The model expects a dataset (e.g., US-Accidents) with columns:

Severity (target variable, e.g., 1-4)
Temperature(F)
Humidity(%)
Pressure(in)
Visibility(mi)
Wind_Speed(mph)
Weather_Condition (categorical, e.g., Clear, Rain, Snow)

The dataset is not included due to size. Download it and place it in the project folder for training.
Files

.gitignore: Excludes large datasets and temporary files.
README.md: Project documentation.
requirements.txt: Python dependencies.
app.py: Streamlit web app code.
train_model.py: Script to train the model and save .pkl files.
model.pkl: Trained RandomForestClassifier model.
feature_names.pkl: List of feature names for the model.
LICENSE: MIT License.

Deployment
Deploy the app using Streamlit Community Cloud:

Sign in with GitHub.
Create a new app, select this repository, and set app.py as the main file.
Deploy to get a public URL.

Testing in Google Colab

Upload accidents.csv, app.py, and train_model.py to Colab.
Install dependencies:!pip install streamlit scikit-learn pandas numpy matplotlib seaborn pyngrok


Run train_model.py to generate .pkl files.
Set up ngrok for Streamlit:!wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
!unzip ngrok-stable-linux-amd64.zip
!./ngrok authtoken YOUR_AUTH_TOKEN


Run the app:!streamlit run app.py &>/dev/null &
!curl -s http://localhost:4040/api/tunnels | python3 -c \
    'import sys, json; print("Public URL:", json.load(sys.stdin)["tunnels"][0]["public_url"])'



License
MIT License (see LICENSE file).
Contributing
Feel free to open issues or submit pull requests to improve the project.
