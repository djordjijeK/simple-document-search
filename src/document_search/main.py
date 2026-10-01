from argparse import ArgumentParser

from document_search.index import InvertedIndex
from document_search.retriever import BM25Retriever, TfIdfRetriever
from document_search.scifact_config import INDEX_PATH
from document_search.scifact_data import download_scifact_data
from document_search.scifact_data_loader import load_scifact_data
from document_search.tokenizer import simple_tokenizer


def search_scifact_data(
    query_id: str = "0", top_k: int = 5, rebuild: bool = False
) -> None:
    if top_k < 1:
        raise ValueError("top_k must be positive")

    # download data
    download_scifact_data()

    # load data
    documents, queries = load_scifact_data()

    if INDEX_PATH.is_file() and not rebuild:
        index = InvertedIndex.load(INDEX_PATH)
    else:
        document_texts = {
            document_id: document.content for document_id, document in documents.items()
        }
        index = InvertedIndex.build(document_texts, simple_tokenizer)
        INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        index.save(INDEX_PATH)

    query = queries[query_id]
    print(f"Indexed documents: {index.document_count}")
    print(f"Query {query_id}: {query}")

    retrievers = [("TF-IDF", TfIdfRetriever(index)), ("BM25", BM25Retriever(index))]

    for name, retriever in retrievers:
        print(f"\n{name}")
        results = retriever.search(query)[:top_k]
        if not results:
            print("No matching documents.")

        for rank, (document_id, score) in enumerate(results, start=1):
            document = documents[document_id]
            print(f"{rank}. [{document_id}] score={score:.4f}")
            print(f"   {document.title}")
            print(f"   {document.abstract}")


def main() -> None:
    argument_parser = ArgumentParser(description="Search local SciFact data")
    argument_parser.add_argument("--query-id", default="0")
    argument_parser.add_argument("--top-k", type=int, default=5)
    argument_parser.add_argument("--rebuild", action="store_true")

    arguments = argument_parser.parse_args()

    search_scifact_data(
        query_id=arguments.query_id,
        top_k=arguments.top_k,
        rebuild=arguments.rebuild,
    )


if __name__ == "__main__":
    main()
