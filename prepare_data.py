import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from open_data import load_data

df = load_data()

df_encoded = pd.get_dummies(df, columns=["Gender", "Workout_Type"], drop_first=True)

x = df_encoded.drop("Experience_Level", axis=1)
y = df_encoded["Experience_Level"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(x)

X_temp, X_test, y_temp, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

print("Data prepared!")
print(f"Train set: {X_train.shape}, {y_train.shape}")
print(f"Validation set: {X_val.shape}, {y_val.shape}")
print(f"Test set: {X_test.shape}, {y_test.shape}")