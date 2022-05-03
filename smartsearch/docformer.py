#!/usr/bin/env python3

from dataset import iter_docs
from transformers import AutoModel, AutoTokenizer, CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer

from pathlib import Path
import re
import torch, numpy as np
import torch.nn.functional as F

def clean(text) -> list[str]:
    textout = text
    if isinstance(textout, str):
        textout = [textout]
    textout = [re.sub(r'\(cid:[0-9]+\)', '', t) for t in textout]
    textout = [re.sub(r'\\u[a-f0-9]+', '', t) for t in textout]
    textout = [re.sub(r'[^\w\s.,!?]',' ', t) for t in textout]
    textout = [re.sub(r'[\s]+',' ', t) for t in textout]
    textout = [re.sub(r'([a-z0-9.,!?])([A-Z])',r'\1 \2', t) for t in textout]
    textout = [re.sub(r'([A-Za-z0-9,.?!])([.,!?])([^A-Za-z0-9,.?!]+)',r'\1\2\3', t) for t in textout]
    textout = [t.strip() for t in textout]
    return textout

#Mean Pooling - Take attention mask into account for correct averaging
#From https://huggingface.co/sentence-transformers/all-mpnet-base-v2
def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Max Pooling - Take the max value over time for every dimension. 
#From https://huggingface.co/sentence-transformers/distilbert-base-nli-max-tokens
def max_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
    return torch.max(token_embeddings, 1)[0]

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

class LongDocformer():
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, text) -> torch.Tensor:
        cleaned_text = clean(text)

        embedings = torch.zeros(len(cleaned_text), self.model.config.hidden_size)
        print(len(cleaned_text))
        n = 16
        for i in range(0, len(cleaned_text), n):
            chunked_text = cleaned_text[i:i + n]
            print(len(chunked_text))
            encoded_input = self.tokenizer(chunked_text, return_tensors='pt', truncation=True, padding=True)
    
            #global mask is [1,0,0,...], in BERT style transformer [CLS] is index 0 which represents rembedding
            global_attention_mask = torch.zeros(encoded_input.attention_mask.shape)
            global_attention_mask[:,0] = 1
            encoded_input['global_attention_mask'] = global_attention_mask
    
            print(encoded_input.input_ids.shape)
            with torch.no_grad():
                output = self.model(**encoded_input)
            print(output[0][:,0].shape)
            chunked_embeddings = mean_pooling(output[0], encoded_input.attention_mask)
            embedings[i:i+n] = chunked_embeddings

        embedings = F.normalize(embedings, p=2, dim=1)
        return embedings

class CLIPDocformer():
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor

    def __call__(self, text) -> torch.Tensor:
        cleaned_text = clean(text)
        if isinstance(cleaned_text, str):
            cleaned_text = [cleaned_text]
        encoded_input = self.processor(text=cleaned_text, images=None, return_tensors='pt', truncation=True, padding=True)

        with torch.no_grad():
            output = self.model.text_model(**encoded_input)
        embeddings = mean_pooling(output[0], encoded_input.attention_mask)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

class SentenceDocformer():
    def __init__(self, model):
        self.model = model

    def __call__(self, text) -> torch.Tensor:
        cleaned_text = clean(text)
        output = self.model.encode(cleaned_text, convert_to_tensor=True, convert_to_numpy=False)
        if output.dim() == 1:
            return output.unsqueeze(0)
        #output is already normalized
        return output


def distilroberta():
    model_name = 'sentence-transformers/all-distilroberta-v1'
    return SentenceDocformer(SentenceTransformer(model_name))

def mpnet_base():
    model_name = 'sentence-transformers/all-mpnet-base-v2'
    return SentenceDocformer(SentenceTransformer(model_name))

def clip_sent():
    # this one will crash from long inputs, TODO submit PR to huggingface
    model_name = 'clip-ViT-B-32'
    return SentenceDocformer(SentenceTransformer(model_name))

def clip():
    model_name = 'openai/clip-vit-base-patch32'
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    return CLIPDocformer(model, processor)

def longformer():
    model_name = 'allenai/longformer-base-4096'
    return LongDocformer(
            AutoModel.from_pretrained(model_name), 
            AutoTokenizer.from_pretrained(model_name))

if __name__ == '__main__':
    embeddings_dir = Path('embeddings')
    if not embeddings_dir.exists():
        embeddings_dir.mkdir()
        
    encoder = longformer()

    for ppdf, ppages in iter_docs():
        pout = Path(f'embeddings/{ppdf.stem}.pt')
        if pout.exists():
            continue
        docs = [ppdf.stem] + ppages
        
        embeddings = encoder(docs)
        torch.save(embeddings, pout)
