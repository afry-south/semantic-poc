#!/usr/bin/env python3

from slate3k import PDF
from pathlib import Path
from multiprocessing import Pool
import re, itertools, traceback
import numpy as np
import torch
import torch.nn.functional as F

class Embeddings():
    def __init__(self, pages = True):
        embeddings_paths = list(iter_embeddings())
        all_embeddings = [torch.load(p).cpu() for p in embeddings_paths]

        self.title_embeddings = np.vstack([e[0] for e in all_embeddings])
        self.page_embeddings = np.vstack([e[1:] for e in all_embeddings])

        self.doc_names = [p.stem for p in embeddings_paths]
        self.nbr_pages = [e.shape[0]-1 for e in all_embeddings]
        self.doc_idx = [[d]*i for d,i in zip(range(len(self.doc_names)), self.nbr_pages)]
        self.doc_idx = [item for sublist in self.doc_idx for item in sublist]
        self.cum_pages = [0]+np.cumsum(np.array(self.nbr_pages))[:-1].tolist()

    def get_doc_page(self, page_idx):
        doc_idx = self.doc_idx[page_idx]
        doc_name = self.doc_names[doc_idx]
        offset = self.cum_pages[doc_idx]
        return doc_name, page_idx - offset

    def get_page_embeddings(self, doc_idx):
        offset = self.cum_pages[doc_idx]
        return [self.page_embeddings[offset+i] for i in range(self.nbr_pages[doc_idx])]

def get_text(name, page):
    pd = Path('docs') / name
    pf = pd.parent / f'{name}.{str(page).zfill(4)}.txt'
    return pf.read_text()

def iter_docs():
    for p in Path('pdfs').glob('*.pdf'):
        txt_files = [pd.read_text() for pd in Path('docs').glob(f'{p.stem}.*.txt')]
        yield p, txt_files

def iter_embeddings():
    for p in Path('embeddings').iterdir():
        yield Path(f'embeddings/{p.stem}.pt')

def process_pdf(pin: Path, pout: Path, join_pages: bool = False):

    def filter_non_utf8(s):
        control_chars = ''.join(map(chr, itertools.chain(range(0x00,0x20), range(0x7f,0xa0))))
        control_char_re = re.compile('[%s]' % re.escape(control_chars))
        s = s.encode('utf-8', 'ignore').decode('utf-8')
        return control_char_re.sub('', s)

    try:
        # Convert PDF to text file
        with pin.open('rb') as fin:
            pdf = PDF(fin)
            if join_pages:
                text = pdf.text(clean = False)
                pout.write_text(filter_non_utf8(text))
            else:
                for i, page in enumerate(pdf):
                    page_pout = pout.with_name(f'{pout.stem}.{str(i).zfill(4)}{pout.suffix}')
                    page_pout.write_text(filter_non_utf8(page))

    except Exception:
        traceback.print_exc()

if __name__ == '__main__':
    docdir = Path('docs')
    print(docdir.resolve())
    if not docdir.exists():
        docdir.mkdir()

    input_paths = list(Path('pdfs').glob('**/*.pdf'))
    output_paths = [p.with_suffix('.txt') for p in input_paths]

    to_process = zip(input_paths, output_paths)

    pool = Pool(processes = 4)
    pool.starmap(process_pdf, to_process)
    pool.close()
