import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from open_data import load_data

# === 1. Load and preprocess === #
df = load_data()
df_encoded = pd.get_dummies(df, columns=["Gender", "Workout_Type"], drop_first=True)

X = df_encoded.drop("Experience_Level", axis=1)
y = df_encoded["Experience_Level"] - 1  # shift to [0, 1, 2]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_temp, X_test, y_temp, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

# === 2. Define hyperparameter grid === #
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 1],
    'colsample_bytree': [0.8, 1]
}

xgb_clf = XGBClassifier(
    use_label_encoder=False,
    eval_metric='mlogloss',
    verbosity=0
)

# === 3. Grid Search === #
grid_search = GridSearchCV(
    estimator=xgb_clf,
    param_grid=param_grid,
    cv=3,
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)

# === 4. Evaluation === #
best_model = grid_search.best_estimator_
val_preds = best_model.predict(X_val)

print("\nBest Parameters:", grid_search.best_params_)
print("\nValidation Accuracy:", accuracy_score(y_val, val_preds))
print("\nClassification Report:")
print(classification_report(y_val, val_preds))

# === 5. Save the best model === #
joblib.dump(best_model, "xgboost_best_model.pkl")
print("\nBest model saved as 'xgboost_best_model.png'")
