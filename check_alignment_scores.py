#Simple python script to check the alignment scores of the generated captions.

import pandas as pd

# Read the alignment scores file
scores_df = pd.read_parquet("./sieve_test/sieve_output/alignment_scores.parquet")

# Display basic statistics
print("\nAlignment Scores Statistics:")
print(scores_df[['clip_score', 'sieve_score']].describe())

# Show a few examples with their captions and scores
print("\nSample Results:")
sample_results = scores_df[['text', 'generated_caption', 'clip_score', 'sieve_score']].head(5)
for _, row in sample_results.iterrows():
    print("\nOriginal:", row['text'])
    print("Generated:", row['generated_caption'])
    print(f"CLIP Score: {row['clip_score']:.4f}")
    print(f"SIEVE Score: {row['sieve_score']:.4f}")
    print("-" * 80)

# Calculate average scores
print("\nAverage Scores:")
print(f"Average CLIP Score: {scores_df['clip_score'].mean():.4f}")
print(f"Average SIEVE Score: {scores_df['sieve_score'].mean():.4f}")