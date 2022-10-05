import pickle as pickle
import pandas as pd
import math
from gensim.models import Word2Vec
import numpy as np
import tqdm

with open('positional_index.pickle', 'rb') as handle:
    index = pickle.load(handle)

archives = [pd.read_excel('IR00_3_11k_News.xlsx'), pd.read_excel('IR00_3_17k_News.xlsx'),
            pd.read_excel('IR00_3_20k_News.xlsx')]
archive = pd.concat(archives, ignore_index=True)
docs = pd.DataFrame(archive)

# embedding
postings_tfidf = dict()
for term in index.keys():
    postings_tfidf[term] = dict()
    nt = len(index[term].keys())
    idf = math.log10(100000 / nt)
    for doc_id in index[term].keys():
        ftd = len(index[term][doc_id])
        tf = 1 + math.log10(ftd)
        postings_tfidf[term][doc_id] = tf * idf

with open("postings_tfid.pickle", "wb") as f:
    pickle.dump(postings_tfidf, f)

model = Word2Vec.load("trained_model/w2v.model")

vectors = dict()
weight_sums = dict()
for term in tqdm.tqdm(index.keys()):
    for doc in index[term].keys():
        if doc not in vectors:
            vectors[doc] = np.zeros(300)
            weight_sums[doc] = 0
        weight_sums[doc] += postings_tfidf[term][doc]
        try:
            vectors[doc] += postings_tfidf[term][doc] * model.wv[term]
        except KeyError:
            print(f"Ignoring: {term}")
            continue

docs_embedding = dict()
for doc, vec in vectors.items():
    docs_embedding[doc] = vec / weight_sums[doc]

with open("embedding.pickle", "wb") as f:
    pickle.dump(docs_embedding, f)
