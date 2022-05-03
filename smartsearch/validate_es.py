#!/usr/bin/env python3
from elasticsearch import Elasticsearch
import torch.nn.functional as F
from dataset import Embeddings
from docformer import BaseDocformer
import smartsearch.elastic

if __name__ == '__main__':
    
    embeddings = Embeddings()
    es = Elasticsearch('http://localhost:9200')

    non_empty = sum(n > 0 for n in embeddings.nbr_pages)

    for k in [1, 5, 10, 25, 50]:
        hits = 0
        for doc_i in range(embeddings.title_embeddings.shape[0]):
            if embeddings.nbr_pages[doc_i] == 0:
                continue
            
            #query_text = embeddings.doc_names[doc_i]
            #query_text = BaseDocformer().clean(query_text)
            #res = smartsearch.index.query_keywords(es, query_text, k)

            query_vector = embeddings.title_embeddings[doc_i]
            res = smartsearch.index.query_encoded(es, query_vector, k=k)

            ids = [int(hit['_id']) for hit in res['hits']['hits']]
            doc_ids = [embeddings.doc_idx[page_id] for page_id in ids]
            if doc_i in doc_ids:
                hits += 1
            
        print(f'top-{k} recall: '+'{:.3f}'.format(hits/non_empty))