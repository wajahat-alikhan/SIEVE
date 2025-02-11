from sentence_transformers import SentenceTransformer, util
from BLIP.models.blip import blip_decoder
from BLIP.models.med import BertConfig
from transformers import BertTokenizer, BertModel
import sys
sys.path.append("open_clip_torch/src/")
import torch

def get_text_encoder():
    config = BertConfig.from_json_file('BLIP/configs/med_config.json')
    config.add_cross_attention = False  # Disable cross-attention
    text_encoder = BertModel(config, add_pooling_layer=False)
    return text_encoder 

def get_text_features(model, text, device='cuda'):
    """Get text features using BLIP's text encoder with visual context"""
    # Create a dummy image tensor of the correct size (batch_size, 3, 224, 224)
    batch_size = len(text)
    dummy_image = torch.zeros(batch_size, 3, 224, 224).to(device)
    
    # Process text through BLIP
    text_input = model.tokenizer(text, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(device)
    
    # Get text features with visual context
    text_features = model.text_encoder(text_input.input_ids, 
                                     attention_mask=text_input.attention_mask,
                                     encoder_hidden_states=model.visual_encoder(dummy_image),
                                     return_dict=True)
    
    # Return the [CLS] token features
    return text_features.last_hidden_state[:, 0, :] 