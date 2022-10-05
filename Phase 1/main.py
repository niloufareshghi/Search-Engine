import pandas as pd
import preprocessing, positional_index, query_response
import pickle

archive = pd.read_excel('IR1_7k_news.xlsx')
docs = pd.DataFrame(archive, columns=['content', 'url', 'title'])

inp = input("Do you want to create/refresh your indexing? (Y/N)\n")
if inp == 'Y':
    tokens = []
    for ind in docs.index:
        tokens.extend(preprocessing.preprocess(docs['content'][ind], ind))
    pos_index = positional_index.pos_indexing(tokens)
    with open('positional_index.pickle', 'wb') as handle:
        pickle.dump(pos_index, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('positional_index.pickle', 'rb') as handle:
    pos_index = pickle.load(handle)

query = input("Please Enter your query:\n")
ans_docs = query_response.response(query, pos_index)
for doc_id in ans_docs:
    print(docs['title'][doc_id])
