#Python script to compare all the scores of the different models, like sentence transformer and CLIP and then compare them.

import pandas as pd
import numpy as np

# Read all score files
st_scores = pd.read_parquet("./sieve_test/sieve_output/st_scores_0_Captioning_base_10000.parquet")
clip_scores = pd.read_parquet("./sieve_test/sieve_output/clip_scores_0_Captioning_base_10000.parquet")
captions_df = pd.read_parquet("./sieve_test/sieve_output/0_Captioning_base_10000.parquet")

# Convert scores from arrays to floats if needed
if isinstance(st_scores['scores'].iloc[0], np.ndarray):
    st_scores['scores'] = st_scores['scores'].apply(lambda x: float(x[0]))
if isinstance(clip_scores['scores'].iloc[0], np.ndarray):
    clip_scores['scores'] = clip_scores['scores'].apply(lambda x: float(x[0]))

# Combine all data
comparison_df = pd.DataFrame({
    'Original_Text': captions_df['text'],
    'Generated_Caption': captions_df['generated_caption'],
    'ST_Score': st_scores['scores'],
    'CLIP_Score': clip_scores['scores']
})

# Display first 5 samples
print("\nFirst 5 samples comparison:")
for _, row in comparison_df.head(5).iterrows():
    print("\nOriginal:", row['Original_Text'])
    print("Generated:", row['Generated_Caption'])
    print(f"Sentence Transformer Score: {float(row['ST_Score']):.4f}")
    print(f"CLIP Score: {float(row['CLIP_Score']):.4f}")
    print("-" * 80)

# Display statistics for both scores
print("\nScore Statistics:")
print("\nSentence Transformer Scores:")
print(comparison_df['ST_Score'].describe().round(4))
print("\nCLIP Scores:")
print(comparison_df['CLIP_Score'].describe().round(4))

# Calculate correlation between scores
correlation = comparison_df['ST_Score'].corr(comparison_df['CLIP_Score'])
print(f"\nCorrelation between ST and CLIP scores: {correlation:.4f}")