import hazm


normalizer = hazm.Normalizer()
stemmer = hazm.Stemmer()


def remove_stop_words(tokens):
    stop_words = hazm.stopwords_list()
    stop_words += ['.', '،', ':', ')', '(', 'کشور', '[', ']', 'https', 'ایر', 'farsnews', '«', '»']
    return [word for word in tokens if word not in stop_words]


def tokenize(content):
    return hazm.word_tokenize(content)


def preprocess(content, doc_id):
    text = normalizer.normalize(content)
    tokens = tokenize(text)
    tokens = remove_stop_words(tokens)
    tokens = [(stemmer.stem(term), doc_id) for term in tokens]
    return tokens

