import pandas as pd
import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image
import tarfile
import io
import sys
import os

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'BLIP'))
sys.path.append("open_clip_torch/src/")

from BLIP.models.blip import blip_decoder
from open_clip.factory import create_model_and_transforms, get_tokenizer

def get_scores(parquet_path, tar_path, device='cuda'):
    # Load models
    # CLIP model
    clip_model, _, preprocess_val = create_model_and_transforms(
        pretrained="openai",
        model_name='ViT-L-14',
        device=device,
        precision="fp32",
        jit=True,
        output_dict=True
    )
    
    # Get CLIP tokenizer
    tokenizer = get_tokenizer('ViT-L-14')
    
    # BLIP model
    blip_model = blip_decoder(pretrained="./weights/model_base_14M.pth", 
                             image_size=224, vit='base').to(device)
    
    # Read data
    df = pd.read_parquet(parquet_path)
    tar = tarfile.open(tar_path, 'r')
    
    clip_scores = []
    sieve_scores = []
    
    # Process in batches
    batch_size = 32
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{len(df)//batch_size + 1}")
        
        # Load and process images
        images = []
        for key in batch['key']:
            img_name = f"{int(key):08d}.jpg"
            img_data = tar.extractfile(img_name).read()
            img = Image.open(io.BytesIO(img_data))
            img = preprocess_val(img)
            images.append(img)
        
        images = torch.stack(images).to(device)
        
        with torch.no_grad():
            # Get CLIP scores
            img_features = clip_model.encode_image(images)
            
            # Tokenize text for CLIP
            text_tokens = tokenizer(batch['text'].tolist()).to(device)
            text_features = clip_model.encode_text(text_tokens)
            
            clip_score = torch.diagonal(img_features @ text_features.T)
            clip_scores.extend(clip_score.cpu().numpy())
            
            # Get SIEVE scores using BLIP
            sieve_score = blip_model.generate(images, sample=True, top_p=0.9)
            sieve_scores.extend([1.0] * len(batch))  # placeholder for now
    
    tar.close()
    
    return clip_scores, sieve_scores

def plot_comparison(clip_scores, sieve_scores):
    plt.figure(figsize=(10, 8))
    plt.scatter(clip_scores, sieve_scores, alpha=0.5)
    plt.xlabel('CLIP Scores')
    plt.ylabel('SIEVE Scores')
    correlation = np.corrcoef(clip_scores, sieve_scores)[0,1]
    plt.title(f'CLIP vs SIEVE Scores\nCorrelation: {correlation:.3f}')
    plt.savefig('./sieve_test/sieve_output/score_comparison.png')
    plt.close()

if __name__ == "__main__":
    parquet_path = "./sieve_test/sieve_output/0_CLIPCAP_10000.parquet"
    tar_path = "./sieve_test/data/00000.tar"
    clip_scores, sieve_scores = get_scores(parquet_path, tar_path)
    plot_comparison(clip_scores, sieve_scores)
    print("Visualization saved as 'score_comparison.png'")