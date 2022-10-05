import random
from math import inf
import numpy as np
from numpy.linalg import norm
import pickle as pickle

with open('positional_index.pickle', 'rb') as handle:
    index = pickle.load(handle)

with open('embedding.pickle', 'rb') as handle:
    docs_embedding = pickle.load(handle)

# k-means
k = int(input("enter the value of k:\n"))


def nearest_center(centers):
    ccls = [set() for _ in centers] # a list of documents appointed to their best matching centers
    for doc, d in docs_embedding.items():
        max_cosine = -inf
        best_center = -1
        for c, center in enumerate(centers):
            cur_cosine = np.dot(d, center) / (norm(d) * norm(center))
            if max_cosine < cur_cosine:
                max_cosine = cur_cosine
                best_center = c
        ccls[best_center].add(doc)
    return ccls


def mean(ccls):
    centers = []
    for ds in ccls:
        cm = np.zeros(300)
        for doc in ds:
            cm += docs_embedding[doc]
        cm = cm / len(ds)
        centers.append(cm)
    return centers


# random centers for k clusters
centers = random.sample(list(docs_embedding), k)
centers = [docs_embedding[c_id] for c_id in centers]
for i in range(20):
    cc_ls = nearest_center(centers)
    centers = mean(cc_ls)

cc_ls = nearest_center(centers)

rss = 0
for i in range(len(cc_ls)):
    for doc in cc_ls[i]:
        d = docs_embedding[doc]
        rss += ((np.dot(centers[i], d) / (norm(d) * norm(centers[i]))) + 1) / 2
print("rss: " + str(rss))

clusters = {'cc_ls': cc_ls, 'centers': centers}

with open("clusters.pickle", "wb") as f:
    pickle.dump(clusters, f)

