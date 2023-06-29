# About the project
This project is my information retrieval course project which is a search engine for retrieving news documents based on users' queries. <br>
The project in implmented using [pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/) and [gensim](https://github.com/RaRe-Technologies/gensim) libraries. 

# Preprocessing

For all query-answering approaches, we need a positional posting list which is extracted in [main.py](https://github.com/niloufareshghi/Search-Engine/blob/main/Phase%201/main.py). <br>
Also for word embedding purposes, a word-to-vector model is constructed in [training.py](https://github.com/niloufareshghi/Search-Engine/blob/main/Phase%202/Bonus/training.py).
# Categorization

In order to answer queries fast enough, we need to categorize documents and queries. <br>
In [knn.py](https://github.com/niloufareshghi/Search-Engine/blob/main/Phase%203/knn.py) a supervised categorizing approach is implemented using the KNN algorithm. <br>
In [clustering.py](https://github.com/niloufareshghi/Search-Engine/blob/main/Phase%203/clustring.py) an unsupervised approach is implemented using the K-means algorithm.

# Query answering

Multiple query-answering approaches are implemented:
+ Simple common word counting approach in [query_response.py](https://github.com/niloufareshghi/Search-Engine/blob/main/Phase%201/query_response.py).
+ An approach based on inner product and tf-idf in [query_champion_list.py](https://github.com/amsen20/search-engine/blob/main/query_champion_list.py) which uses champion lists in order to speed up.
+ A word embedding-based approach with inner product criteria in [query_response.py](https://github.com/niloufareshghi/Search-Engine/blob/main/Phase%202/Bonus/query_response.py).
+ A fast version of the word embedding approach in [query_clustering.py](https://github.com/niloufareshghi/Search-Engine/blob/main/Phase%203/query_clustering.py) which uses clusters in order to speed up.
+ A category-aware approach in [query_knn.py](https://github.com/niloufareshghi/Search-Engine/blob/main/Phase%203/query_knn.py) that the user enters the category along with the query itself.
