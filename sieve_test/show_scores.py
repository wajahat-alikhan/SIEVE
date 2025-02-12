import pandas as pd

# Read the score files
clip_scores = pd.read_parquet("./sieve_test/sieve_output/clip_scores_0_Captioning_base_10000.parquet")
sieve_scores = pd.read_parquet("./sieve_test/sieve_output/scores_0_Captioning_base_10000.parquet")

# Combine scores into one dataframe
comparison_df = pd.DataFrame({
    'CLIP_Score': clip_scores['scores'],
    'SIEVE_Score': sieve_scores['scores']
})

# Display first 10 samples
print("\nFirst 10 samples comparison:")
print(comparison_df.head(10).round(4))

# Display some basic statistics
print("\nBasic Statistics:")
print(comparison_df.describe().round(4))

# Calculate correlation
correlation = comparison_df['CLIP_Score'].corr(comparison_df['SIEVE_Score'])
print(f"\nCorrelation between CLIP and SIEVE scores: {correlation:.4f}")