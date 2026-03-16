"""Quick script to check the CSV structure"""
import pandas as pd
import os

DATA_DIR = os.path.join("..", "00_Data", "raw")

# Read with multi-row header
df = pd.read_csv(os.path.join(DATA_DIR, "Daily.csv"),
                 sep=";",
                 encoding="utf-8-sig",
                 header=[0, 1],
                 nrows=5)

print("Column names (MultiIndex):")
print(df.columns)
print("\nColumn levels:")
print("Level 0:", df.columns.get_level_values(0).tolist())
print("Level 1:", df.columns.get_level_values(1).tolist())
print("\nFirst few rows:")
print(df.head())
