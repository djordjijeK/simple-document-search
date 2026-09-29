import json
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from document_search.tokenizer import TOKENIZERS

DocumentFrequency = dict[str, int]


@dataclass
class InvertedIndex:
    term_to_document_term_frequency: dict[str, DocumentFrequency]
    document_to_total_document_terms: dict[str, int]
    document_count: int
    tokenizer: Callable[[str], list[str]]

    @classmethod
    def build(
        cls, documents: dict[str, str], tokenizer: Callable[[str], list[str]]
    ) -> "InvertedIndex":
        term_to_document_term_frequency: dict[str, DocumentFrequency] = {}
        document_to_total_document_terms: dict[str, int] = {}

        for document_id, document in documents.items():
            terms = tokenizer(document)
            document_to_total_document_terms[document_id] = len(terms)

            for term, frequency in Counter(terms).items():
                term_documents = term_to_document_term_frequency.setdefault(term, {})
                term_documents[document_id] = frequency

        return cls(
            term_to_document_term_frequency=term_to_document_term_frequency,
            document_to_total_document_terms=document_to_total_document_terms,
            document_count=len(documents),
            tokenizer=tokenizer,
        )

    @property
    def vocabulary(self) -> set[str]:
        return set(self.term_to_document_term_frequency)

    @property
    def average_document_length(self) -> float:
        if self.document_count == 0:
            return 0.0

        return sum(self.document_to_total_document_terms.values()) / self.document_count

    def term_frequency(self, term: str, document_id: str) -> int:
        term_documents = self.term_to_document_term_frequency.get(term, {})
        return term_documents.get(document_id, 0)

    def document_frequency(self, term: str) -> int:
        term_documents = self.term_to_document_term_frequency.get(term, {})
        return len(term_documents)

    def documents_with(self, term: str) -> tuple[str, ...]:
        term_documents = self.term_to_document_term_frequency.get(term, {})
        return tuple(sorted(term_documents))

    def save(self, path: str | Path) -> None:
        tokenizer_name = next(
            (
                name
                for name, implementation in TOKENIZERS.items()
                if implementation is self.tokenizer
            ),
            None,
        )
        if tokenizer_name is None:
            raise ValueError("This tokenizer has no saved configuration")

        data = {
            "version": 1,
            "tokenizer": tokenizer_name,
            "term_to_document_term_frequency": self.term_to_document_term_frequency,
            "document_to_total_document_terms": self.document_to_total_document_terms,
            "document_count": self.document_count,
        }

        Path(path).write_text(
            json.dumps(data, ensure_ascii=False, indent=4, sort_keys=True),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "InvertedIndex":
        data = json.loads(Path(path).read_text(encoding="utf-8"))

        if data.get("version") != 1:
            raise ValueError("Unsupported index version")

        if data.get("tokenizer") not in TOKENIZERS:
            raise ValueError("Unsupported index tokenizer")

        document_to_total_document_terms = data["document_to_total_document_terms"]

        if data["document_count"] != len(document_to_total_document_terms):
            raise ValueError(
                "Corrupted index content. Index document count does not match its data"
            )

        return cls(
            term_to_document_term_frequency=data["term_to_document_term_frequency"],
            document_to_total_document_terms=document_to_total_document_terms,
            document_count=data["document_count"],
            tokenizer=TOKENIZERS[data["tokenizer"]],
        )
