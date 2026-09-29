from collections import Counter
from math import log, sqrt

from document_search.index import InvertedIndex

DocumentScore = tuple[str, float]


class TfIdfRetriever:
    def __init__(self, index: InvertedIndex):
        self._inverted_index = index
        self._term_to_document_weight = {
            term: {
                document_id: self._term_weight(term, frequency)
                for document_id, frequency in document_frequencies.items()
            }
            for term, document_frequencies in index.term_to_document_term_frequency.items()
        }
        self._document_to_document_norm = self._calculate_document_norms()

    def search(self, query: str) -> list[DocumentScore]:
        query_term_frequencies = Counter(self._inverted_index.tokenizer(query))
        query_term_weights = {
            query_term: self._term_weight(query_term, query_term_frequency)
            for query_term, query_term_frequency in query_term_frequencies.items()
        }

        query_norm = sqrt(
            sum(term_weight**2 for term_weight in query_term_weights.values())
        )

        if query_norm == 0:
            return []

        dot_products: dict[str, float] = {}

        for query_term, query_term_weight in query_term_weights.items():
            document_weights = self._term_to_document_weight.get(query_term, {})
            for document_id, document_weight in document_weights.items():
                dot_products[document_id] = dot_products.get(document_id, 0.0) + (
                    document_weight * query_term_weight
                )

        results = []

        for document_id, dot_product in dot_products.items():
            document_norm = self._document_to_document_norm[document_id]

            if document_norm > 0 and dot_product > 0:
                score = dot_product / (query_norm * document_norm)
                results.append((document_id, score))

        return sorted(results, key=lambda result: (-result[1], result[0]))

    def _calculate_document_norms(self) -> dict[str, float]:
        document_to_squared_norm = {
            document_id: 0.0
            for document_id in self._inverted_index.document_to_total_document_terms
        }

        for document_weights in self._term_to_document_weight.values():
            for document_id, weight in document_weights.items():
                document_to_squared_norm[document_id] += weight**2

        return {
            document_id: sqrt(squared_norm)
            for document_id, squared_norm in document_to_squared_norm.items()
        }

    def _term_weight(self, term: str, term_frequency: int) -> float:
        if term_frequency == 0:
            return 0.0

        return (1 + log(term_frequency)) * self._inverse_document_frequency(term)

    def _inverse_document_frequency(self, term: str) -> float:
        document_frequency = self._inverted_index.document_frequency(term)
        if document_frequency == 0:
            return 0.0

        return log(self._inverted_index.document_count / document_frequency)
