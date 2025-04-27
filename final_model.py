import pandas as pd
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from xgboost import XGBClassifier
from open_data import load_data

# === 1. Load and preprocess === #
df = load_data()
df_encoded = pd.get_dummies(df, columns=["Gender", "Workout_Type"], drop_first=True)

X = df_encoded.drop("Experience_Level", axis=1)
y = df_encoded["Experience_Level"] - 1

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_temp, X_test, y_temp, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

# === 2. Define Optuna objective === #
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'eval_metric': 'mlogloss',
        'use_label_encoder': False,
        'verbosity': 0,
    }

    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    return acc

# === 3. Run Optuna Study === #
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50, timeout=300)

print("\nBest trial:")
print("  Accuracy:", study.best_value)
print("  Params:", study.best_params)

# === 4. Train best model and evaluate === #
best_model = XGBClassifier(**study.best_params, use_label_encoder=False, eval_metric='mlogloss', verbosity=0)
best_model.fit(X_train, y_train)
val_preds = best_model.predict(X_val)

print("\nFinal Optuna-Tuned Model Accuracy:", accuracy_score(y_val, val_preds))
print("\nClassification Report:")
print(classification_report(y_val, val_preds))

# === 5. Save confusion matrix as .png === #
plt.figure(figsize=(6, 4))
disp = ConfusionMatrixDisplay.from_predictions(y_val, val_preds, display_labels=['Beginner', 'Intermediate', 'Advanced'], cmap='Blues')
plt.title("Confusion Matrix: Optuna-Tuned XGBoost")
plt.tight_layout()
plt.savefig("optuna_xgboost_confusion_matrix.png")
print("\nConfusion matrix saved as 'optuna_xgboost_confusion_matrix.png'")
