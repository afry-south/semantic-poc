import streamlit as st
from elasticsearch import Elasticsearch
from elastic import query_encoded, query_keywords, query_prefix
from docformer import mpnet_base
from collections import OrderedDict
import time

@st.cache(allow_output_mutation=True)
def encoder():
    return mpnet_base()

st.caption(':memo: The demo is based on a free collection of 1477 PDFs on computer science.')

with st.expander('Configure'):
    mode = st.radio(
        "Select search mode",
        ('Semantic (ANN-similarity)', 'Terms (TF-IDF)', 'Prefix'))

    k = st.slider('Results: ', min_value=5, max_value=200, value=10, step=5)

    group = st.checkbox('Group documents', value=True)

st.header('Semantic search demo')

query = st.text_input('', placeholder='Enter query ...', key='query')

if query:
    start = time.process_time()
    es = Elasticsearch('http://localhost:9200')
    
    if mode == 'Semantic (ANN-similarity)':
        res = query_encoded(es, query, encoder=encoder(), k=k)
    elif mode == 'Prefix':
        res = query_prefix(es, query, k=k)
    else:
        res = query_keywords(es, query, k=k)

    encode_time = '{0:.2f}'.format(time.process_time() - start)

    total = res['hits']['total']['value']

    st.caption(f'Query took {encode_time}s, {total} results.')

    retrieved_docs = set()

    dct = OrderedDict()
    for page_hit in res['hits']['hits']:
        doc_title = page_hit['fields']['doc_title'][0]
        dct.setdefault(doc_title, []).append(page_hit)

    st.markdown('---')

    def truncate(str, n):
        return (str[:n] + ' ...') if len(str) > n else str

    list_results = list(dct.values()) if group else [[l] for l in res['hits']['hits']]

    for page_hits in list_results:
        doc_title = truncate(page_hits[0]['fields']['doc_title'][0], 75)
        total_pages = page_hits[0]['fields']['total_pages'][0]
        
        st.markdown(f'##### {doc_title}')
        
        for i,page_hit in enumerate(page_hits):
            score = '{0:.2f}'.format(page_hit['_score'])
            page_id = int(page_hit['_id'])
            text = page_hit['fields']['text'][0]
            truncated_text = truncate(text, 128)
            page = page_hit['fields']['page_nr'][0]
            indent = '> ' if i>0 else ''

            st.caption(f'{indent}Page {page} / {total_pages}, {len(text)} characters,   relevance: {score}')
            if len(truncated_text) != len(text) and i==0:
                #with st.expander(f'{indent}{truncated_text}', expanded=False):
                st.write(f'{indent}{text}')
            else:
                st.write(f'{indent}{text}')
            #st.markdown('#####')

        st.markdown('##')
