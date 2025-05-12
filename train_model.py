import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv('accidents.csv')
df.dropna(inplace=True)

# Select features
features = ['Severity', 'Temperature(F)', 'Humidity(%)', 'Pressure(in)', 
            'Visibility(mi)', 'Wind_Speed(mph)', 'Weather_Condition']
df = df[features]

# Convert categorical data
df = pd.get_dummies(df, columns=['Weather_Condition'], drop_first=True)

# Save feature names
feature_names = df.drop('Severity', axis=1).columns.tolist()
with open('feature_names.pkl', 'wb') as f:
    joblib.dump(feature_names, f)

# Split dataset
X = df.drop('Severity', axis=1)
y = df['Severity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=20, random_state=42)
model.fit(X_train, y_train)

# Save model
with open('model.pkl', 'wb') as f:
    joblib.dump(model, f)
