import hazm


normalizer = hazm.Normalizer()
stemmer = hazm.Stemmer()
stop_words = hazm.stopwords_list()


def tokenize(query):
    return hazm.word_tokenize(query)


def response(query, pos_index):
    query = normalizer.normalize(query)
    tokens = tokenize(query)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [stemmer.stem(term) for term in tokens]
    if len(tokens) == 1:
        d = [(doc, len(pos_index[query][doc])) for doc in pos_index[query].keys()]
        d.sort(key=lambda x: x[1], reverse=True)
        return [e for e in d[:10]]
    else:
        ls = [set(pos_index[tokens[i]].keys()) for i in range(1, len(tokens))]
        shared_docs = set(pos_index[tokens[0]].keys()).intersection(*ls)
        best_fit = []
        for doc in shared_docs:
            count = 0
            for pos in pos_index[tokens[0]][doc]:
                flag = 1
                for i in range(1, len(tokens)):
                    if pos + i not in pos_index[tokens[i]][doc]:
                        flag = 0
                        break
                if flag == 1:
                    count += 1
            if count > 0:
                best_fit.append((doc, count))
        best_fit.sort(key=lambda x: x[1], reverse=True)
        ls = [set(pos_index[tokens[i]].keys()) for i in range(1, len(tokens))]
        all_docs = set(pos_index[tokens[0]].keys()).union(*ls)
        relev = all_docs - shared_docs
        others = []
        for doc in relev:
            rank = 0
            for token in tokens:
                if doc in pos_index[token].keys():
                    rank += len(pos_index[token][doc])
            others.append((doc, rank))
        others.sort(key=lambda x: x[1], reverse=True)

        final_list = []
        for e in best_fit:
            final_list.append(e)
        for e in others[:10]:
            final_list.append(e)
        return final_list
