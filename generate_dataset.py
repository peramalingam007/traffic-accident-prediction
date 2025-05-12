import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
n_samples = 10000
data = {
    'Severity': np.random.randint(1, 5, n_samples),
    'Temperature(F)': np.random.uniform(-20, 120, n_samples),
    'Humidity(%)': np.random.uniform(0, 100, n_samples),
    'Pressure(in)': np.random.uniform(28, 31, n_samples),
    'Visibility(mi)': np.random.uniform(0, 10, n_samples),
    'Wind_Speed(mph)': np.random.uniform(0, 50, n_samples),
    'Weather_Condition': np.random.choice(
        ['Clear', 'Rain', 'Snow', 'Fog', 'Cloudy', 'Thunderstorm', 'Haze'], 
        n_samples, 
        p=[0.4, 0.2, 0.1, 0.1, 0.1, 0.05, 0.05]
    )
}

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('accidents.csv', index=False)
print(f"Synthetic dataset saved as 'accidents.csv' with {n_samples} rows (~{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB)")
