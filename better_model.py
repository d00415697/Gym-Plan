import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from open_data import load_data

# Load and preprocess data
df = load_data()
df_encoded = pd.get_dummies(df, columns=["Gender", "Workout_Type"], drop_first=True)

X = df_encoded.drop("Experience_Level", axis=1)
y = df_encoded["Experience_Level"]

# No strict need to scale for Random Forest, but let's keep consistent
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_temp, X_test, y_temp, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

# Train Random Forest
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Evaluate
train_preds = model.predict(X_train)
val_preds = model.predict(X_val)

print("Random Forest Results")
print(f"Training Accuracy: {accuracy_score(y_train, train_preds):.4f}")
print(f"Validation Accuracy: {accuracy_score(y_val, val_preds):.4f}")

print("\n Classification Report (Validation):")
print(classification_report(y_val, val_preds))
