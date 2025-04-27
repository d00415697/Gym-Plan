import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import keras

def load_data():
    file_path = "gym_members_exercise_tracking.csv"
    try:
        df = pd.read_csv(file_path)
        print("Dataset loaded successfully!")
        print(f"shape of dataset: {df.shape}")
        return df
    except FileNotFoundError:
        print("can't locate file.")
        return None
if __name__ == "__main__":
    load_data()