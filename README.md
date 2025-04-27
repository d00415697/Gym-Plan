# Gym Members Experience Prediction

## 📚 Project Overview
This project builds a machine learning pipeline to predict **gym members' experience levels** (Beginner, Intermediate, Advanced) using their demographic, health, and workout data.

We developed multiple models including Logistic Regression, Random Forest, XGBoost, and an Optuna-tuned final XGBoost model, achieving over **90% validation accuracy**.

---

## 📊 Dataset Description
The dataset includes 973 entries with the following features:

- **Demographics:** Age, Gender (one-hot encoded)
- **Physical Attributes:** Height, Weight, BMI, Fat Percentage
- **Workout Metrics:** Workout Type (one-hot encoded), Session Duration, Calories Burned, Workout Frequency
- **Health Metrics:** Resting BPM, Max Heart Rate, Avg Heart Rate
- **Other:** Water Intake per Day
- **Target:** Experience Level (Beginner, Intermediate, Advanced)

---

## 🛠 Installation


## 🚀 How to Run

### 1. Explore the data
```bash
python explore_data.py
```
Generates distributions, categorical plots, and a correlation heatmap.

### 2. Prepare and split the data
```bash
python prepare_data.py
```
Splits into 60% training, 20% validation, 20% testing.

### 3. Run baseline models
```bash
python first_model.py  # Logistic Regression
python better_model.py  # Random Forest
```

### 4. Train refined models
```bash
python refined_model.py  # Random Forest & XGBoost (GridSearch)
```

### 5. Tune final model with Optuna
```bash
python optuna_xgboost_tuner.py
```
Outputs:
- Best hyperparameters
- Final Validation Scores
- Confusion matrix saved as `optuna_xgboost_confusion_matrix.png`

---

## 📈 Model Building Process

- **First Model:** Logistic Regression (84.6% Validation Accuracy)
- **Better Model:** Random Forest (87.2% Validation Accuracy)
- **Refined Model:** Tuned Random Forest and XGBoost (up to 91.8% Validation Accuracy)
- **Final Model:** Optuna-Tuned XGBoost (90.3% Validation Accuracy, best overall generalization)

Confusion matrix showed strong prediction across all classes, especially Advanced.

---

## 📷 Visualizations
- Feature Distributions
- Correlation Heatmap
- Confusion Matrix (Optuna XGBoost)

All plots are saved as `.png` files.

---

## 🧰 Technologies Used
- Python 3.10+
- Pandas
- Numpy
- Scikit-learn
- XGBoost
- Optuna
- Matplotlib
- Seaborn

---

## 🛤️ Future Improvements
- Ensemble Voting Classifier combining Logistic, RF, and XGBoost
- SHAP explainability plots
- Early stopping and calibration
- Deployment with Streamlit or Flask
- Export model to TensorFlow Lite for mobile applications

---

## 👨‍💻 Credits
Developed by Deray Lowe as part of the CS-4320 Machine Learning course project.

Special thanks to the course instructors and Optuna open-source contributors!
