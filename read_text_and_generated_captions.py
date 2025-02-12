import pandas as pd

# Read the parquet file
df = pd.read_parquet("./sieve_test/sieve_output/0_Captioning_base_10000.parquet")

# Display first few rows
print("\nFirst 5 rows:")
print(df.head())

# Display basic information
print("\nDataset Info:")
print(df.info())

# Check total number of captions
print(f"\nTotal number of captions: {len(df)}")

# Check if there are any missing values
print("\nMissing values:")
print(df.isnull().sum())