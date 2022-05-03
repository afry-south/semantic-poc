#!/usr/bin/env python3

from elasticsearch import Elasticsearch
from dataset import Embeddings, get_text

def create_index(es):
    request_body = {
        'mappings' : {
            'properties' : {
                'doc_title' : {
                    'type' : 'text',
                    'index': False
                    #'fields' : {'keyword' : {'type' : 'keyword','ignore_above' : 256}}
                },
                'page_nr' : {'type' : 'integer', 'index': False},
                'total_pages' : {'type' : 'integer', 'index': False},
                'embedding' : {
                    'type': 'dense_vector',
                    'dims': 768,
                    'index': True,
                    'similarity': 'dot_product' 
                },
                'text' : {
                    'type' : 'text',
                    'index': True
                    #'fields' : {'keyword' : {'type' : 'keyword','ignore_above' : 256}}
                },
            }
        }
	}
    es.indices.create(index = 'doc_index', body = request_body)

def index_docs(es, embeddings):
    #TODO use es.bulk

    for page_i in range(len(embeddings.page_embeddings)):
        doc_i = embeddings.doc_idx[page_i]
        name,page_nr = embeddings.get_doc_page(page_i)
        total_pages = embeddings.nbr_pages[doc_i]
        embedding = embeddings.page_embeddings[page_i]
        text = get_text(name, page_nr)
        
        doc = {
            'doc_title': name,
            'page_nr': page_nr,
            'total_pages': total_pages,
            'embedding': embedding,
            'text': text
        }
        es.index(index='doc_index', id=page_i, document=doc)

def query_encoded(es, query, encoder=None, k=5):
    if isinstance(query, str):
        query = encoder(query)
    query = query.squeeze().tolist()
    
    knn = {
        'field': 'embedding',
        'query_vector': query,
        'k': k,
        'num_candidates': 200
    }
    fields = ['doc_title', 'page_nr', 'total_pages', '_id', 'text']
    return es.knn_search(knn=knn, index='doc_index', fields=fields, source=False)

def query_keywords(es, query_text, k=5):
    query = {
        'simple_query_string': {
            'query': query_text,
            'fields': ['text']
        }
    }        
    fields = ['doc_title', 'page_nr', 'total_pages', '_id', 'text']

    return es.search(query=query, index='doc_index', fields=fields, source=False, size=k)

def query_prefix(es, query_text, k=5):
    # some other potential solutions:
    #"query": {
    #  "match_phrase_prefix": {
    #    "message": {
    #      "query": "quick brown f",
    #        "max_expansions": 10
    #    }
    #  }
    #}
    #"query": {
    #"bool": {
    #    "should": [
    #    { "prefix": { "name.keyword": "Eli" } },
    #    { "fuzzy": { "name.keyword": { "value": "Eli", "fuzziness": 2, "prefix_length": 0 } } }
    #    ]
    #}
    #}

    query = {
        "prefix" : { "text" : query_text }
    }
    fields = ['doc_title', 'page_nr', 'total_pages', '_id', 'text']
    return es.search(query=query, index='doc_index', fields=fields, source=False, size=k)

if __name__ == '__main__':
    es = Elasticsearch('http://localhost:9200')
    create_index(es)

    embeddings = Embeddings()

    index_docs(es, embeddings)
