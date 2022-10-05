def pos_indexing(tokens):
    pos_index = dict()
    for pos, token in enumerate(tokens):
        term = token[0]
        doc_id = token[1]
        if term not in pos_index:
            pos_index[term] = {}
        if doc_id not in pos_index[term]:
            pos_index[term][doc_id] = set()
        pos_index[term][doc_id].add(pos)
    return pos_index
