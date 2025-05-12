AI-Driven Traffic Accident Severity Prediction

This project uses a RandomForestClassifier to predict traffic accident severity based on weather and environmental conditions. The model is deployed as a Streamlit web app, allowing users to input features like temperature, humidity, and weather condition to predict accident severity.

Repository Structure

graph TD
    A[traffic-accident-prediction] --> B[.gitignore]
    A --> C[README.md]
    A --> D[requirements.txt]
    A --> E[app.py]
    A --> F[train_model.py]
    A --> G[generate_dataset.py]
    A --> H[accidents.csv]
    A --> I[LICENSE]

Workflow

graph TD
    A[Develop in Google Colab] --> B[Run generate_dataset.py]
    B --> C[Generate accidents.csv]
    C --> D[Run train_model.py]
    D --> E[Generate model.pkl & feature_names.pkl]
    E --> F[Test app.py with Streamlit]
    F --> G[Push Files to GitHub]
    G --> H[Deploy to Streamlit Community Cloud]
    H --> I[Update Code]
    I --> G

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

Dataset

The model uses a synthetic dataset (accidents.csv, ~2-5MB, 10,000 rows) to meet the <20MB size constraint. Columns:





Severity (target variable, 1-4)



Temperature(F) (temperature in Fahrenheit)



Humidity(%) (relative humidity percentage)



Pressure(in) (atmospheric pressure in inches)



Visibility(mi) (visibility distance in miles)



Wind_Speed(mph) (wind speed in miles per hour)



Weather_Condition (categorical: Clear, Rain, Snow, Fog, Cloudy, Thunderstorm, Haze)

Dataset Generation:





The repository includes accidents.csv. To regenerate it, run generate_dataset.py:

python generate_dataset.py

Model Generation

The model.pkl and feature_names.pkl files are not included in the repository to keep it lightweight. To generate them:





Run train_model.py after generating accidents.csv:

python train_model.py



This creates model.pkl (~48MB) and feature_names.pkl (~1KB) locally.

Note: model.pkl is large but regenerated on demand, avoiding GitHub size limits.

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

Files





.gitignore: Excludes temporary files, model.pkl, and feature_names.pkl.



README.md: Project documentation.



requirements.txt: Python dependencies.



app.py: Streamlit web app code, regenerates .pkl files if missing.



train_model.py: Script to train model and save .pkl files.



generate_dataset.py: Script to generate synthetic accidents.csv.



accidents.csv: Synthetic dataset (~2-5MB).



LICENSE: MIT License.

Testing in Google Colab





Upload app.py, train_model.py, generate_dataset.py, and (optionally) accidents.csv to Colab.



Install dependencies:

!pip install streamlit>=1.20.0 scikit-learn>=1.2.0 pandas>=1.5.0 numpy>=1.23.0 matplotlib>=3.6.0 seaborn>=0.12.0 joblib>=1.2.0 pyngrok



Generate dataset (if needed):

!python generate_dataset.py



Generate model:

!python train_model.py



Set up ngrok for Streamlit:

!wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
!unzip ngrok-stable-linux-amd64.zip
!./ngrok authtoken YOUR_AUTH_TOKEN



Run the app:

!streamlit run app.py &>/dev/null &
!curl -s http://localhost:4040/api/tunnels | python3 -c \
    'import sys, json; print("Public URL:", json.load(sys.stdin)["tunnels"][0]["public_url"])'

Deployment

Deploy the app using Streamlit Community Cloud:





Sign in with GitHub.



Create a new app, select this repository, and set app.py as the main file.



The app automatically regenerates model.pkl and feature_names.pkl if missing.



Deploy to get a public URL.

License

MIT License (see LICENSE file).

Contributing

Feel free to open issues or submit pull requests to improve the project.

Note

The synthetic dataset ensures a small size (<20MB) for ease of use in Colab. For production, consider sampling real data from the US-Accidents dataset.
