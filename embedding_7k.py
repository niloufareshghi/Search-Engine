import pickle as pickle
import pandas as pd
import math
from gensim.models import Word2Vec
import numpy as np
import tqdm

with open('../Phase1/positional_index.pickle', 'rb') as handle:
    index = pickle.load(handle)

archive = pd.read_excel('IR1_7k_news.xlsx')
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

model = Word2Vec.load("../trained_model/w2v.model")

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
