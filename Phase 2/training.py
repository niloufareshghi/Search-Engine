import multiprocessing
import hazm
from gensim.models import Word2Vec
import pandas as pd

cores = multiprocessing.cpu_count()

archive = pd.read_excel('../IR1_7k_news.xlsx')
docs = pd.DataFrame(archive, columns=['content', 'url', 'title'])

training_data = []
for ind in docs.index:
    normalizer = hazm.Normalizer()
    stemmer = hazm.Stemmer()
    stop_words = hazm.stopwords_list()
    text = normalizer.normalize(docs['content'][ind])
    tokens = hazm.word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [stemmer.stem(token) for token in tokens]
    training_data.append(tokens)

w2v_model = Word2Vec(
    min_count=1,
    window=5,
    vector_size=300,
    alpha=0.03,
    workers=cores-1
)

w2v_model.build_vocab(training_data)
w2v_model.train(training_data, total_examples=w2v_model.corpus_count, epochs=20)
w2v_model.save("w2v_300d.model")

