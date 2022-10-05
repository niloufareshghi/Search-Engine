import random

import numpy as np
import pickle as pickle
import pandas as pd
import tqdm
from numpy.linalg import norm

with open('embedding.pickle', 'rb') as handle:
    embedding_50k = pickle.load(handle)

with open('Phase2/embedding.pickle', 'rb') as handle:
    embedding_7k = pickle.load(handle)

with open('clusters.pickle', 'rb') as handle:
    clusters = pickle.load(handle)

centers = clusters['centers']
cc_ls = clusters['cc_ls']

archives = [pd.read_excel('IR00_3_11k_News.xlsx'), pd.read_excel('IR00_3_17k_News.xlsx'),
            pd.read_excel('IR00_3_20k_News.xlsx')]
archive = pd.concat(archives, ignore_index=True)
docs = pd.DataFrame(archive)

catgorize = dict()
for doc_id, doc in tqdm.tqdm(embedding_7k.items()):
    c_id = list(range(len(centers)))
    c_id.sort(key=lambda x: ((np.dot(centers[x], doc) / (norm(doc) * norm(centers[x]))) + 1) / 2,
              reverse=True)
    i = c_id[0]
    docs_classified = list(cc_ls[i])
    random.shuffle(docs_classified)
    docs_classified = docs_classified[:1000]
    docs_classified.sort(key=lambda dc: ((np.dot(embedding_50k[dc], doc) / (norm(doc) * norm(embedding_50k[dc]))) + 1) / 2, reverse=True)
    count_category = dict()
    for dc in docs_classified[:15]:
        category = docs["topic"][dc]
        if category not in count_category:
            count_category[category] = 0
        count_category[category] += 1
    count_max = -1
    best_category = None
    for ctg, count in count_category.items():
        if count > count_max:
            count_max = count
            best_category = ctg
    catgorize[doc_id] = best_category

with open("categorized.pickle", 'wb') as handle:
    pickle.dump(catgorize, handle)
