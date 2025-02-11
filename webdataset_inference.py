import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import webdataset as wds
import io
import tarfile

def visualize_samples(tar_path, parquet_path, num_samples=5):
    # Read both results
    df_clipcap = pd.read_parquet(parquet_path)
    df_blip = pd.read_parquet(parquet_path.replace('CLIPCAP', 'Captioning_base'))
    
    # Open tar file
    tar = tarfile.open(tar_path, 'r')
    
    # Get random samples
    samples = df_clipcap.sample(num_samples)
    
    # Create a figure with subplots
    fig = plt.figure(figsize=(15, 4*num_samples))
    
    for idx, (index, row) in enumerate(samples.iterrows()):
        # Format image name with 8 digits (00000044.jpg instead of 44.jpg)
        img_name = f"{int(row['key']):08d}.jpg"
        
        try:
            # Get image from tar
            img_data = tar.extractfile(img_name).read()
            img = Image.open(io.BytesIO(img_data))
            
            # Create subplot
            ax = plt.subplot(num_samples, 2, idx*2 + 1)
            ax.imshow(img)
            ax.axis('off')
            
            # Get BLIP scores for this sample
            blip_row = df_blip.loc[df_blip['key'] == row['key']]
            
            # Add text information
            text = f"Original: {row['text']}\n\n"
            text += f"Generated (CLIPCAP): {row['generatedtext']}\n"
            if 'clip_scores' in row:
                text += f"CLIP Score: {row['clip_scores']:.3f}\n"
            if not blip_row.empty and 'cap_scores' in blip_row.columns:
                text += f"SIEVE Score: {blip_row['cap_scores'].iloc[0]:.3f}\n"
            
            plt.subplot(num_samples, 2, idx*2 + 2)
            plt.text(0, 0.5, text, wrap=True)
            plt.axis('off')
            
        except KeyError as e:
            print(f"Could not find image: {img_name}")
            continue
    
    plt.tight_layout()
    plt.savefig('./sieve_test/sieve_output/sample_analysis_with_scores.png', 
                bbox_inches='tight', dpi=300)
    plt.close()
    
    tar.close()

if __name__ == "__main__":
    tar_path = "./sieve_test/data/00000.tar"
    parquet_path = "./sieve_test/sieve_output/0_CLIPCAP_10000.parquet"
    visualize_samples(tar_path, parquet_path)
    print("Visualization saved as 'sample_analysis_with_scores.png'")