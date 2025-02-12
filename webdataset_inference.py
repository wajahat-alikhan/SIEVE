import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import webdataset as wds
import io
import tarfile
import os
import pyarrow as pa
import pyarrow.parquet as pq
import argparse
from BLIP.models.blip import blip_decoder
import torch
from torchvision import transforms
import open_clip
from open_clip import create_model_and_transforms, get_tokenizer

def get_args_parser():
    parser = argparse.ArgumentParser('WebDataset inference', add_help=False)
    # New arguments from README
    parser.add_argument('--data_dir', default='', type=str,
                        help='path to directory containing tar files')
    parser.add_argument('--captioning', action='store_true',
                        help='enable captioning mode')
    parser.add_argument('--clipcap', action='store_true',
                        help='compute SIEVE and CLIPScore alignment scores')
    parser.add_argument('--model_url', default='', type=str,
                        help='url to download model weights')
    parser.add_argument('--save_all_captions', action='store_true',
                        help='save all generated captions instead of just the best one')
    
    # Keep existing arguments
    parser.add_argument('--model_type', default='Captioning_base', type=str,
                        help='model type: Captioning_base or CLIPCAP')
    parser.add_argument('--tar_data', default='', type=str,
                        help='path to tar file')
    parser.add_argument('--output_dir', default=None, type=str,
                        help='path to save results')
    parser.add_argument('--visualize', action='store_true',
                        help='visualize samples')
    return parser.parse_args()

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
        for _, row in batch.iterrows():
            img_data = tar.extractfile(f"{int(i):08d}.jpg").read()
            img = Image.open(io.BytesIO(img_data))
            img = preprocess_val(img)
            images.append(img)
        
        images = torch.stack(images).to(device)
        
        with torch.no_grad():
            # Get CLIP scores
            img_features = clip_model.encode_image(images)
            text_tokens = tokenizer(batch['generated_caption'].tolist()).to(device)
            text_features = clip_model.encode_text(text_tokens)
            clip_score = torch.diagonal(img_features @ text_features.T)
            clip_scores.extend(clip_score.cpu().numpy())
            
            # Get SIEVE scores using BLIP
            sieve_score = blip_model.compute_similarity(images, batch['text'].tolist())
            sieve_scores.extend(sieve_score.cpu().numpy())
    
    tar.close()
    return clip_scores, sieve_scores

def visualize_samples(tar_path, parquet_path):
    # Only try to read the parquet file if it exists
    if os.path.exists(parquet_path):
        df_clipcap = pd.read_parquet(parquet_path)
        df_blip = pd.read_parquet(parquet_path.replace('CLIPCAP', 'Captioning_base'))
        
        # Open tar file
        tar = tarfile.open(tar_path, 'r')
        
        # Get random samples
        samples = df_clipcap.sample(5)
        
        # Create a figure with subplots
        fig = plt.figure(figsize=(15, 4*5))
        
        for idx, (index, row) in enumerate(samples.iterrows()):
            # Format image name with 8 digits (00000044.jpg instead of 44.jpg)
            img_name = f"{int(row['key']):08d}.jpg"
            
            try:
                # Get image from tar
                img_data = tar.extractfile(img_name).read()
                img = Image.open(io.BytesIO(img_data))
                
                # Create subplot
                ax = plt.subplot(5, 2, idx*2 + 1)
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
                
                plt.subplot(5, 2, idx*2 + 2)
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
    else:
        print(f"Parquet file {parquet_path} not found. Skipping visualization.")

def run_inference(args):
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                           (0.26862954, 0.26130258, 0.27577711))
    ])
    
    # Load appropriate model based on mode
    if args.captioning:
        model = blip_decoder(pretrained="./weights/model_base_14M.pth", image_size=224, vit='base')
        model.eval()
        model = model.to('cuda')
    elif args.clipcap:
        # For now, we'll still use BLIP for computing scores
        model = blip_decoder(pretrained="./weights/model_base_14M.pth", image_size=224, vit='base')
        model.eval()
        model = model.to('cuda')
    else:
        raise ValueError("Must specify either --captioning or --clipcap")
    
    # Get specific tar file path
    tar_path = os.path.join(args.data_dir, "00000.tar")
    dataset = wds.WebDataset(tar_path).decode("pil").to_tuple("jpg", "txt")
    
    # Lists to store results
    all_captions = []
    all_texts = []
    
    if args.clipcap:
        # Use the existing get_scores function
        clip_scores, sieve_scores = get_scores(
            parquet_path=os.path.join(args.output_dir, "0_Captioning_base_10000.parquet"),
            tar_path=os.path.join(args.data_dir, "00000.tar"),
            device='cuda'
        )
        
        # Read the original captions file
        caption_df = pd.read_parquet(os.path.join(args.output_dir, "0_Captioning_base_10000.parquet"))
        
        # Save scores in separate files (following the pattern in show_scores.py)
        clip_df = pd.DataFrame({'scores': clip_scores})
        sieve_df = pd.DataFrame({'scores': sieve_scores})
        
        clip_df.to_parquet(os.path.join(args.output_dir, "clip_scores_0_Captioning_base_10000.parquet"))
        sieve_df.to_parquet(os.path.join(args.output_dir, "scores_0_Captioning_base_10000.parquet"))
        print("Saved alignment scores")
        
        return os.path.join(args.output_dir, "scores_0_Captioning_base_10000.parquet")
    
    elif args.captioning:
        # Process each image
        with torch.no_grad():
            for i, (img, txt) in enumerate(dataset):
                # Preprocess image
                image = transform(img).unsqueeze(0).to('cuda')
                
                # Generate caption using BLIP
                caption = model.generate(image, 
                                      sample=True,
                                      num_beams=1,
                                      max_length=20, 
                                      min_length=5,
                                      repetition_penalty=1.1)
                all_captions.append(caption[0])
                all_texts.append(txt)
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1} images")
        
        # Save results based on mode
        output_file = os.path.join(args.output_dir, f"0_Captioning_base_{len(all_captions)}.parquet")
        df = pd.DataFrame({'text': all_texts, 'generated_caption': all_captions})
        df.to_parquet(output_file)
        print(f"Saved results to {output_file}")
        return output_file

if __name__ == "__main__":
    args = get_args_parser()
    
    # Run inference first
    tar_path = args.tar_data
    parquet_path = run_inference(args)
    
    # Then try to visualize (if file exists)
    if args.visualize:
        visualize_samples(tar_path, parquet_path)
    print("Visualization saved as 'sample_analysis_with_scores.png'")