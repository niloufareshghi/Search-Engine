import pandas as pd
from Phase1 import preprocessing, positional_index
import pickle as pickle

archives = [pd.read_excel('IR00_3_11k_News.xlsx'), pd.read_excel('IR00_3_17k_News.xlsx'),
            pd.read_excel('IR00_3_20k_News.xlsx')]
archive = pd.concat(archives, ignore_index=True)
docs = pd.DataFrame(archive)
tokens = []
for ind in docs.index:
    new_t = preprocessing.preprocess(docs['content'][ind], ind)
    tokens.extend(new_t)
pos_index = positional_index.pos_indexing(tokens)
with open('positional_index.pickle', 'wb') as handle:
    pickle.dump(pos_index, handle, protocol=pickle.HIGHEST_PROTOCOL)
