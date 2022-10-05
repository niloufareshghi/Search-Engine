import time

import math
import sys

import hazm
import pandas as pd
import pickle as pickle

with open('Phase1/positional_index.pickle', 'rb') as handle:
    index = pickle.load(handle)

archive = pd.read_excel('Phase2/IR1_7k_news.xlsx')
docs = pd.DataFrame(archive)

with open("categorized.pickle", 'rb') as handle:
    categorize = pickle.load(handle)

n = int(input("how many docs should be returned?"))

postings = dict()
length = dict()
for term in index.keys():
    postings[term] = dict()
    for doc_id in index[term].keys():
        ftd = len(index[term][doc_id])
        w = 1 + math.log10(ftd)
        postings[term][doc_id] = w
        if doc_id not in length:
            length[doc_id] = 0
        length[doc_id] += w ** 2

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
    ctg = input("Please enter the category:\n")
    start = time.time()
    if query == 'exit':
        sys.exit()
    tokens = query_handling(query)
    tf = dict()
    documents = set()
    for t in set(tokens):
        ftq = tokens.count(t)
        tf[t] = 1 + math.log10(ftq)
        for d in index[t].keys():
            if ctg == "unsure" or ctg == categorize[d]:
                documents.add(d)

    scores = dict()
    for t in set(tokens):
        if t not in index:
            continue
        for doc in documents:
            if doc not in scores:
                scores[doc] = 0
            if doc in postings[t]:
                scores[doc] += postings[t][doc] * tf[t]

    for d in scores:
        if d in length and length[d] > 0:
            scores[d] = scores[d] / math.sqrt(length[d])

    returned_docs = list(scores)
    returned_docs = list(filter(lambda x: scores[x] > 0, returned_docs))
    returned_docs.sort(key=lambda x: scores[x], reverse=True)
    returned_docs = returned_docs[:n]
    end = time.time()
    print(f"time elapsed: {end - start}")
    for d in returned_docs:
        print(scores[d], " :: ",  docs['title'][d], " :: ", d)
