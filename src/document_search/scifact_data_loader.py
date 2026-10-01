import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from document_search.scifact_config import (
    CORPUS_FILENAME,
    DATASET_PATH,
    QUERIES_FILENAME,
)


@dataclass(frozen=True)
class SciFactDocument:
    title: str
    abstract: str

    @property
    def content(self) -> str:
        return f"{self.title}\n\n{self.abstract}".strip()


def __read_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error


def load_scifact_data() -> tuple[dict[str, SciFactDocument], dict[str, str]]:
    path = DATASET_PATH

    documents: dict[str, SciFactDocument] = {}
    for record in __read_jsonl(path / CORPUS_FILENAME):
        document_id = record["_id"]
        if document_id in documents:
            raise ValueError(f"Duplicate document id: {document_id}")

        documents[document_id] = SciFactDocument(
            title=record["title"],
            abstract=record["text"],
        )

    queries: dict[str, str] = {}
    for record in __read_jsonl(path / QUERIES_FILENAME):
        query_id = record["_id"]
        if query_id in queries:
            raise ValueError(f"Duplicate query ID: {query_id}")

        queries[query_id] = record["text"]

    return documents, queries


if __name__ == "__main__":
    documents, queries = load_scifact_data()
    print(f"{len(documents)} documents, {len(queries)} queries")

    first_id, first_document = next(iter(documents.items()))
    print(first_id, first_document)
