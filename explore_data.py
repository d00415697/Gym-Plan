import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from open_data import load_data

df = load_data()

print("\n Basic Info:", df.info())
print("\n Summary Statistics:", df.describe())

def plot_distribution(column, bins = 30):
    plt.figure(figsize=(7, 4))
    sns.histplot(df[column], kde=True, bins=bins)
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"plot_{column.replace(' ', '_')}.png")
    plt.close()

    
for col in ["Age", "Calories_Burned", "Session_Duration (hours)", "BMI"]:
    plot_distribution(col)
    
def plot_categorical(column):
    plt.figure(figsize=(6,4))
    sns.countplot(data=df, x=column, order=df[column].value_counts().index)
    plt.title(f"Count of {column}")
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"plot_{column.replace(' ', '_')}.png")
    plt.close()

    
for col in ["Gender", "Workout_Type", "Experience_Level"]:
    plot_categorical(col)
    
plt.figure(figsize=(12, 8))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="Workout_Frequency (days/week)", y="Calories_Burned")
plt.title("Calories Burned vs. Workout Frequency")
plt.xlabel("Workout Frequency (days/week)")
plt.ylabel("Calories Burned")
plt.tight_layout()
plt.show()