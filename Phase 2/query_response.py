import sys
from gensim.models import Word2Vec
import numpy as np
from numpy.linalg import norm
import pickle5 as pickle
import math
import hazm
import pandas as pd


with open('../positional_index.pickle', 'rb') as handle:
    index = pickle.load(handle)

archive = pd.read_excel('../IR1_7k_news.xlsx')
docs = pd.DataFrame(archive, columns=['content', 'url', 'title'])


postings_tfidf = dict()
for term in index.keys():
    postings_tfidf[term] = dict()
    nt = len(index[term].keys())
    idf = math.log10(10000/nt)
    for doc_id in index[term].keys():
        ftd = len(index[term][doc_id])
        tf = 1 + math.log10(ftd)
        postings_tfidf[term][doc_id] = tf*idf


my_w2v_model = Word2Vec.load("w2v_300d.model")
gen_w2v_model = Word2Vec.load("w2v_150k_hazm_300_v2.model")

user_model = input("which trained model?(my/gen)\n")
model = None
if user_model == 'my':
    model = my_w2v_model
if user_model == 'gen':
    model = gen_w2v_model

vectors = dict()
weight_sums = dict()
for term in index.keys():
    for doc in index[term].keys():
        if doc not in vectors:
            vectors[doc] = np.zeros(300)
            weight_sums[doc] = 0
        try:
            vectors[doc] += postings_tfidf[term][doc] * model.wv[term]
        except KeyError:
            print(f"Ignoring: {term}")
            continue
        weight_sums[doc] += postings_tfidf[term][doc]

docs_embedding = []
for doc in vectors:
    docs_embedding.append(vectors[doc]/weight_sums[doc])


def query_handling(q):
    normalizer = hazm.Normalizer()
    stemmer = hazm.Stemmer()
    stop_words = hazm.stopwords_list()
    q = normalizer.normalize(q)
    terms = hazm.word_tokenize(q)
    terms = [word for word in terms if word not in stop_words]
    terms = [stemmer.stem(term) for term in terms]
    return terms


while True:
    query = input("Please enter your query:\n")
    if query == 'exit':
        sys.exit()
    tokens = query_handling(query)
    q_vec = np.zeros(300)
    weights_sum = 0
    q_embedding = []
    for t in tokens:
        if t not in postings_tfidf:
            continue
        ftq = tokens.count(t)
        tf = 1 + math.log10(ftq)
        nt = len(postings_tfidf[t].keys())
        idf = math.log10(10000/ nt)
        q_vec += model.wv[t] * tf * idf
        weights_sum += tf * idf
        q_embedding = q_vec/weights_sum

    scores = dict()
    for i, doc in enumerate(vectors.keys()):
        if doc not in scores.keys():
            scores[doc] = 0
        scores[doc] = ((np.dot(docs_embedding[i], q_embedding) / (norm(docs_embedding[i]) * norm(q_embedding))) + 1) / 2
    sorted_tuples = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    k_best = [key for key, val in sorted_tuples][0:5]

    for doc in k_best:
        print(docs['title'][doc])
