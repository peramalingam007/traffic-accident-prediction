import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pickle

# Load synthetic dataset
df = pd.read_csv('accidents.csv')

# Display basic info
print("Dataset head:")
print(df.head())
print("\nWeather Condition values:")
print(df['Weather_Condition'].unique())

# Drop rows with missing values
df.dropna(inplace=True)

# Select relevant features
features = ['Severity', 'Temperature(F)', 'Humidity(%)', 'Pressure(in)', 
            'Visibility(mi)', 'Wind_Speed(mph)', 'Weather_Condition']
df = df[features]

# Convert categorical data to numeric
df = pd.get_dummies(df, columns=['Weather_Condition'], drop_first=True)

# Save feature names for Streamlit app
feature_names = df.drop('Severity', axis=1).columns.tolist()
with open('feature_names.pkl', 'wb') as f:
    pickle.dump(feature_names, f)

# Split dataset into X and y
X = df.drop('Severity', axis=1)
y = df['Severity']

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix.png')
plt.show()
