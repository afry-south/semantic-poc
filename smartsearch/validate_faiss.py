#!/usr/bin/env python3
import faiss, autofaiss
from dataset import Embeddings
from pathlib import Path


if __name__ == '__main__':
    
    pdir = Path('faiss_index')
    pidx = str(pdir / 'knn.index')
    pinfo = str(pdir / 'infos.json')
    
    embeddings = Embeddings()

    if not pdir.exists():
        print('building index')
        pdir.mkdir()
        index = autofaiss.build_index(embeddings.page_embeddings, save_on_disk=True, index_path=pidx, index_infos_path=pinfo)[0]
    else:
        index = faiss.read_index(pidx)

    non_empty = sum(n > 0 for n in embeddings.nbr_pages)

    for k in [1, 5, 10, 25, 50]:
        # searches for all titles in the dataset in batch
        _, indices = index.search(embeddings.title_embeddings, k)
        hits = 0
        for i in range(indices.shape[0]):
            if embeddings.nbr_pages[i] == 0:
                continue
            if i in [embeddings.doc_idx[j] for j in indices[i]]:
                hits += 1

        print(f'top-{k} recall: '+'{:.3f}'.format(hits/non_empty))