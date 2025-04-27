import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from open_data import load_data

# Load and preprocess
df = load_data()
df_encoded = pd.get_dummies(df, columns=["Gender", "Workout_Type"], drop_first=True)

X = df_encoded.drop("Experience_Level", axis=1)
y = df_encoded["Experience_Level"] - 1

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_temp, X_test, y_temp, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

# Hyperparameter Grid for Random Forest
rf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
}

rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=3, n_jobs=-1)
rf_grid.fit(X_train, y_train)

print("Best Random Forest Params:", rf_grid.best_params_)

# Evaluate Tuned Random Forest
rf_preds = rf_grid.predict(X_val)
print("\nRandom Forest (Tuned) Validation Accuracy:", accuracy_score(y_val, rf_preds))
print(classification_report(y_val, rf_preds))

# Train XGBoost
xgb_model = XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, use_label_encoder=False, eval_metric='mlogloss')
xgb_model.fit(X_train, y_train)

xgb_preds = xgb_model.predict(X_val)
print("\nXGBoost Validation Accuracy:", accuracy_score(y_val, xgb_preds))
print(classification_report(y_val, xgb_preds))
