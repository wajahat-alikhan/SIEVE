#This script shows the scores of the Sentence Transformer for the first 5 samples.

import pandas as pd
import numpy as np

# Read the score files
st_scores = pd.read_parquet("./sieve_test/sieve_output/st_scores_0_Captioning_base_10000.parquet")
captions_df = pd.read_parquet("./sieve_test/sieve_output/0_Captioning_base_10000.parquet")

# Convert scores from arrays to floats if needed
if isinstance(st_scores['scores'].iloc[0], np.ndarray):
    st_scores['scores'] = st_scores['scores'].apply(lambda x: float(x[0]))

# Combine scores with captions
comparison_df = pd.DataFrame({
    'Original_Text': captions_df['text'],
    'Generated_Caption': captions_df['generated_caption'],
    'ST_Score': st_scores['scores']
})

# Display first 5 samples
print("\nFirst 5 samples comparison:")
for _, row in comparison_df.head(5).iterrows():
    print("\nOriginal:", row['Original_Text'])
    print("Generated:", row['Generated_Caption'])
    print(f"Sentence Transformer Score: {float(row['ST_Score']):.4f}")
    print("-" * 80)

# Display basic statistics
print("\nScore Statistics:")
print(comparison_df['ST_Score'].describe().round(4))

# Calculate average score
print(f"\nAverage ST Score: {comparison_df['ST_Score'].mean():.4f}")