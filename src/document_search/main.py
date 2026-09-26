from math import log, sqrt


def tf(document, term):
    term_frequency = document.split().count(term)
    return 1 + log(term_frequency) if term_frequency else 0


def idf(documents, term):
    document_frequency = sum(term in text.split() for text in documents.values())
    return log(len(documents) / document_frequency) if document_frequency else 0


def tfidf_weight(documents, document, term):
    return tf(document, term) * idf(documents, term)


def vector_norm(documents, document):
    sum_of_squares = sum(
        tfidf_weight(documents, document, term.strip()) ** 2
        for term in set(document.split())
    )

    return sqrt(sum_of_squares)


def dot_product(documents, query, document):
    return sum(
        tfidf_weight(documents, document, term) * tfidf_weight(documents, query, term)
        for term in set(query.split())
    )


def cosine_similarity(documents, document, query):
    query_vector_norm = vector_norm(documents, query)
    document_vector_norm = vector_norm(documents, document)

    if query_vector_norm == 0 or document_vector_norm == 0:
        return 0.0

    return dot_product(documents, query, document) / (
        query_vector_norm * document_vector_norm
    )


def search(query, documents):
    scores = []

    for document_id, document in documents.items():
        score = cosine_similarity(documents, document, query)

        if score > 0:
            scores.append((document_id, score))

    return sorted(scores, key=lambda score: (-score[1], score[0]))


if __name__ == "__main__":
    query = "gene disease"

    documents = {
        "d1": "gene therapy treats disease",
        "d2": "gene gene expression predicts disease",
        "d3": "climate model predicts warming",
    }

    for document_id, score in search(query, documents):
        print(document_id, f"{score:.6f}")
