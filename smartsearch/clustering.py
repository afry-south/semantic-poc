#!/usr/bin/env python3

import umap,hdbscan
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

# this file is based on the blog post:
# https://towardsdatascience.com/topic-modeling-with-bert-779f7db187e6

if __name__ == "__main__":
    pathlist = Path('embeddings').glob('*.pt')
    documents = []
    print('loading data')
    for p in pathlist:
        document = {}
        documents.append(document)
        document['name'] = p.stem
        document['embedding'] = torch.load(p)
        document['text'] = Path(f'docs/{p.stem}.txt').read_text()
    data = list(map(lambda d: d['text'], documents))
    embeddings = torch.stack(list(map(lambda d: d['embedding'], documents)))
    
    print('reducing dimensions')
    umap_embeddings = umap.UMAP(n_neighbors=10, 
                                n_components = 5, 
                                metric = 'cosine').fit_transform(embeddings)
    print('clustering')
    cluster = hdbscan.HDBSCAN(min_cluster_size=10,
                              metric='euclidean',
                              cluster_selection_method='eom').fit(umap_embeddings)


    # Prepare data
    umap_data = umap.UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
    result = pd.DataFrame(umap_data, columns=['x', 'y'])
    result['labels'] = cluster.labels_

    # Visualize clusters
    fig, ax = plt.subplots(figsize=(20, 10))
    outliers = result.loc[result.labels == -1, :]
    clustered = result.loc[result.labels != -1, :]
    plt.scatter(outliers.x, outliers.y, color='#BDBDBD', s=0.05)
    plt.scatter(clustered.x, clustered.y, c=clustered.labels, s=0.05, cmap='hsv_r')
    plt.colorbar()
    plt.show()


    print('creating frame for topic importance')
    docs_df = pd.DataFrame(data, columns=["Doc"])
    docs_df['Topic'] = cluster.labels_
    docs_df['Doc_ID'] = range(len(docs_df))
    docs_per_topic = docs_df.groupby(['Topic'], as_index = False).agg({'Doc': ' '.join})

    def c_tf_idf(documents, m, ngram_range=(1, 1)):
        count = CountVectorizer(ngram_range=ngram_range, stop_words="english").fit(documents)
        t = count.transform(documents).toarray()
        w = t.sum(axis=1)
        tf = np.divide(t.T, w)
        sum_t = t.sum(axis=0)
        idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)
        tf_idf = np.multiply(tf, idf)

        return tf_idf, count
  
    print('calculating ctf-idf')
    tf_idf, count = c_tf_idf(docs_per_topic.Doc.values, m=len(data))


    def extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20):
        words = count.get_feature_names()
        labels = list(docs_per_topic.Topic)
        tf_idf_transposed = tf_idf.T
        indices = tf_idf_transposed.argsort()[:, -n:]
        top_n_words = {label: [(words[j], tf_idf_transposed[i][j]) for j in indices[i]][::-1] for i, label in enumerate(labels)}
        return top_n_words
    
    def extract_topic_sizes(df):
        topic_sizes = (df.groupby(['Topic'])
                         .Doc
                         .count()
                         .reset_index()
                         .rename({"Topic": "Topic", "Doc": "Size"}, axis='columns')
                         .sort_values("Size", ascending=False))
        return topic_sizes

    print('extracting top-n words')
    top_n_words = extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20)
    topic_sizes = extract_topic_sizes(docs_df);
    ix = topic_sizes['Topic'] >= 0
    top_topics = topic_sizes[ix]['Topic'][0:5].values.tolist()
    print(topic_sizes)
    print(top_n_words.keys())
    for i in range(0,len(top_topics)):
        print(f'topic {i}')
        print(top_topics[i])
        print(top_n_words[top_topics[i]][:10])
