
import pandas as pd

# Path to the Amazon reviews dataset
file_path = "dataset/Cell_Phones_and_Accessories_5.json"

print("Loading dataset...")

# Read the JSON file (one JSON record per line)
df = pd.read_json(file_path, lines=True)

print("\nDataset loaded successfully!")

# Display basic information
print("\nFirst 5 records:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nRating distribution:")
print(df["overall"].value_counts().sort_index())
