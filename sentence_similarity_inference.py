# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import fsspec
from sentence_transformers import SentenceTransformer
import torch
import torch.nn
from torch.utils.data import Dataset, DataLoader
import argparse
import time
import os
import glob
import multiprocessing as mp
from multiprocessing import Pool, current_process, Queue
from tqdm import tqdm
import datetime
import re
import sys
sys.path.append("open_clip_torch/src/")
from open_clip.factory import create_model_and_transforms, get_tokenizer

# Set the start method to 'spawn'
mp.set_start_method('spawn', force=True)

queue = Queue()
NUM_CAPTIONS = 8

REMOVE_PHRASES_LONG = ["an image of", "a photo of", "an icon of", "an illustration of",
                                                "a template of", "a thumbnail of", "a vector of", 
                                                 "photo stock", "stock photo", 
                                                 "a photo", "an image", "an icon", "an illustration", "a template", "a thumbnail",
                                                 "image", "photo", "icon", "illustration", "template", "vector", "thumbnail",
                                                 "free", "print", "sale", "quot", "png", "jpeg", "jpg"]

REMOVE_PHRASES = ['an image of', "a photo of", "stock photo", "photo stock", "a photo", "an image", "image", "photo"]

print(f'The list of phrases to exclude from captions and generated captions: {REMOVE_PHRASES}')
   
def remove_phrases(sentences, phrases_to_remove=REMOVE_PHRASES):
    modified_sentences = []
    # ignore case sensitivity
    phrases_to_remove = [re.compile(phrase, re.IGNORECASE) for phrase in phrases_to_remove]
    for sentence in sentences:
        for phrase in phrases_to_remove:
            sentence = phrase.sub('', sentence)
        modified_sentences.append(sentence)
    return modified_sentences

def read_parquet(url):
    # First, read the file to see what columns we have
    df = pd.read_parquet(url)
    print(f"Available columns in {url}:", df.columns.tolist())
    
    # Map the columns we have to what we need
    keys = df['key'].tolist() if 'key' in df.columns else [str(i) for i in range(len(df))]
    uids = [str(i) for i in range(len(df))]  # Create sequential UIDs
    
    # Get captions - original text
    captions = df['text'].tolist() if 'text' in df.columns else []
    
    # Get generated text
    if 'generatedtext' in df.columns:
        generated_text = df['generatedtext'].tolist()
    elif 'generated_caption' in df.columns:
        generated_text = df['generated_caption'].tolist()
    else:
        generated_text = captions  # fallback to original captions
    
    # Wrap generated text in list since we only have one generation
    generated_text_list = [generated_text]
    
    # Use the only generation as best
    best_generated_text = generated_text
    
    # Use placeholder scores if not available
    cap_scores = df['cap_scores'].tolist() if 'cap_scores' in df.columns else [1.0] * len(df)
    
    # Use original captions as raw
    raw_captions = captions
    
    return uids, keys, captions, generated_text_list, best_generated_text, cap_scores, raw_captions

def get_args_parser():
    parser = argparse.ArgumentParser('Sentence similarity inference', add_help=False)
    parser.add_argument('--parquet_dir', default='', type=str,
                        help='path to parquet files directory')
    parser.add_argument('--output_dir', default=None, type=str,
                        help='path to save results')
    parser.add_argument('--sentence_transformer_text_encoder', action='store_true',
                        help='use sentence transformer text encoder')
    parser.add_argument('--clip_text_encoder', action='store_true',
                        help='use CLIP text encoder')
    parser.add_argument('--blip_text_encoder', action='store_true',
                        help='use BLIP text encoder')
    parser.add_argument('--per_device_batch_size', default=32, type=int,
                        help='batch size per device')
    parser.add_argument('--device', default='cuda', type=str,
                        help='device to use for inference')
    return parser.parse_args()

def process_batch(batch_text, batch_generated_text, model, device):
    with torch.no_grad():
        text_embedd = model.encode(batch_text, convert_to_tensor=True, device=device)
        generated_embedd = model.encode(batch_generated_text, convert_to_tensor=True, device=device)
        text_embedd = text_embedd.unsqueeze(1)
        generated_embedd = generated_embedd.unsqueeze(1)
        cosine_sim = torch.nn.functional.cosine_similarity(text_embedd, generated_embedd, dim=2)
    return cosine_sim.cpu().numpy()

def main():
    args = get_args_parser()
    print(args)
    
    # Load appropriate model based on arguments
    if args.sentence_transformer_text_encoder:
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        model.to(args.device)
    elif args.clip_text_encoder:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-L-14',
            pretrained='openai',
            device=args.device
        )
        tokenizer = open_clip.get_tokenizer('ViT-L-14')
    elif args.blip_text_encoder:
        from BLIP.models.blip import blip_decoder
        model = blip_decoder(pretrained="./weights/model_base_14M.pth", 
                           image_size=224, vit='base')
        model.to(args.device)
        model.eval()
    else:
        raise ValueError("Must specify one of --sentence_transformer_text_encoder, --clip_text_encoder, or --blip_text_encoder")
    
    parquet_files = glob.glob(os.path.join(args.parquet_dir, "*.parquet"))
    
    for parquet_file in parquet_files:
        if os.path.basename(parquet_file).startswith(('scores_', 'clip_scores_', 'st_scores_')):
            print(f"Skipping score file: {parquet_file}")
            continue
            
        print(f"Processing {parquet_file}")
        df = pd.read_parquet(parquet_file)
        print(f"Available columns: {df.columns.tolist()}")
        
        # Check if this file has generated text
        if not any(col in df.columns for col in ['generatedtext', 'generated_caption']):
            print(f"Skipping {parquet_file} - no generated text column found")
            continue
        
        # Prepare data
        captions = df['text'].tolist()
        
        # Get generated text from appropriate column
        if 'generatedtext' in df.columns:
            generated_text = df['generatedtext'].tolist()
        else:
            generated_text = df['generated_caption'].tolist()
        
        # Process batches using GPU
        scores = []
        batch_size = args.per_device_batch_size
        for i in range(0, len(captions), batch_size):
            batch_text = captions[i:i + batch_size]
            batch_generated_text = generated_text[i:i + batch_size]
            
            if args.sentence_transformer_text_encoder:
                batch_scores = process_batch(batch_text, batch_generated_text, model, args.device)
            elif args.clip_text_encoder:
                with torch.no_grad():
                    text_tokens = tokenizer(batch_text).to(args.device)
                    text_features = model.encode_text(text_tokens)
                    gen_text_tokens = tokenizer(batch_generated_text).to(args.device)
                    gen_text_features = model.encode_text(gen_text_tokens)
                    text_features = text_features / text_features.norm(dim=1, keepdim=True)
                    gen_text_features = gen_text_features / gen_text_features.norm(dim=1, keepdim=True)
                    similarity = torch.sum(text_features * gen_text_features, dim=1)
                    batch_scores = similarity.cpu().numpy()
            else:  # BLIP text encoder
                with torch.no_grad():
                    # BLIP uses bert tokenizer
                    text_input = model.tokenizer(batch_text, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(args.device)
                    gen_text_input = model.tokenizer(batch_generated_text, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(args.device)
                    
                    # Get text features using BLIP's text decoder
                    text_features = model.text_decoder(text_input.input_ids, attention_mask=text_input.attention_mask)[0][:, 0, :]
                    gen_text_features = model.text_decoder(gen_text_input.input_ids, attention_mask=gen_text_input.attention_mask)[0][:, 0, :]
                    
                    # Normalize features
                    text_features = text_features / text_features.norm(dim=1, keepdim=True)
                    gen_text_features = gen_text_features / gen_text_features.norm(dim=1, keepdim=True)
                    
                    # Compute similarity
                    similarity = torch.sum(text_features * gen_text_features, dim=1)
                    batch_scores = similarity.cpu().numpy()
            
            scores.extend(batch_scores)
            print(f"Processed batch {i//batch_size + 1}/{(len(captions) + batch_size - 1)//batch_size}, total scores: {len(scores)}")
        
        # Save results with appropriate prefix
        if args.clip_text_encoder:
            prefix = "clip_"
        elif args.blip_text_encoder:
            prefix = "blip_"
        else:
            prefix = "st_"
            
        output_file = os.path.join(args.output_dir if args.output_dir else os.path.dirname(parquet_file),
                                 f"{prefix}scores_{os.path.basename(parquet_file)}")
        pd.DataFrame({'scores': scores}).to_parquet(output_file)
        print(f"Saved scores to {output_file}")

if __name__ == '__main__':
    main()
            

