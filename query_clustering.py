import sys
import time

from gensim.models import Word2Vec
import numpy as np
from numpy.linalg import norm
import pickle as pickle
import math
import hazm
import pandas as pd

with open('positional_index.pickle', 'rb') as handle:
    index = pickle.load(handle)

archives = [pd.read_excel('IR00_3_11k_News.xlsx'), pd.read_excel('IR00_3_17k_News.xlsx'),
            pd.read_excel('IR00_3_20k_News.xlsx')]
archive = pd.concat(archives, ignore_index=True)
docs = pd.DataFrame(archive)

with open('embedding.pickle', 'rb') as handle:
    docs_embedding = pickle.load(handle)

with open('clusters.pickle', 'rb') as handle:
    clusters = pickle.load(handle)

centers = clusters['centers']
cc_ls = clusters['cc_ls']

with open('postings_tfid.pickle', 'rb') as handle:
    tfidf = pickle.load(handle)

model = Word2Vec.load("trained_model/w2v.model")


def query_handling(q):
    normalizer = hazm.Normalizer()
    stemmer = hazm.Stemmer()
    stop_words = hazm.stopwords_list()
    q = normalizer.normalize(q)
    terms = hazm.word_tokenize(q)
    terms = [word for word in terms if word not in stop_words]
    terms = [stemmer.stem(term) for term in terms]
    return terms


b = int(input("what is the value of b?\n"))
n = int(input("how many docs should be returned?\n"))

while True:
    query = input("Please enter your query:\n")
    start = time.time()
    if query == 'exit':
        sys.exit()
    tokens = query_handling(query)
    q_vec = np.zeros(300)
    weights_sum = 0
    for t in set(tokens):
        ftq = tokens.count(t)
        tf = 1 + math.log10(ftq)
        nt = len(tfidf[t].keys())
        idf = math.log10(100000 / nt)
        weights_sum += tf * idf
        try:
            q_vec += model.wv[t] * tf * idf
        except KeyError:
            print(f"Ignoring: {t}")

    q_embedding = q_vec/weights_sum

    c_id = list(range(len(clusters)))
    c_id.sort(key=lambda x: (np.dot(centers[x], q_embedding) / (norm(q_embedding) * norm(centers[x]))),
              reverse=True)
    c_id = c_id[:b]

    returned_docs = []
    for i in c_id:
        returned_docs += cc_ls[i]
    returned_docs = list(set(returned_docs))

    returned_docs.sort(key=lambda d: np.dot(docs_embedding[d], q_embedding) / (norm(q_embedding) * norm(docs_embedding[d])),
                       reverse=True)
    returned_docs = returned_docs[:n]

    end = time.time()
    print(f"time elapsed: {end-start}")
    for doc in returned_docs:
        print(docs['url'][doc])
