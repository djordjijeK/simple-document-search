from argparse import ArgumentParser
from pathlib import Path

from document_search.index import InvertedIndex
from document_search.retriever import TfIdfRetriever
from document_search.tokenizer import simple_tokenizer


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--load-index", type=Path)
    parser.add_argument("--save-index", type=Path)
    args = parser.parse_args()

    documents = {
        "d1": (
            "gene therapy delivers a healthy gene to cells and may treat "
            "inherited disease in patients"
        ),
        "d2": (
            "gene gene expression changes when cells face stress and altered "
            "gene activity can predict disease risk"
        ),
        "d3": (
            "a disease outbreak spread through several cities while public "
            "health teams tracked new cases daily"
        ),
        "d4": (
            "researchers found a gene that controls how plants respond to "
            "heat and limited water"
        ),
        "d5": (
            "climate models estimate future warming from changing emissions "
            "and ocean temperatures across many regions"
        ),
    }

    inverted_index = (
        InvertedIndex.load(args.load_index)
        if args.load_index is not None
        else InvertedIndex.build(documents, simple_tokenizer)
    )
    if args.save_index is not None:
        inverted_index.save(args.save_index)

    tf_idf_retriever = TfIdfRetriever(inverted_index)
    for document_id, score in tf_idf_retriever.search(args.query):
        print(document_id, f"{score:.6f}")


if __name__ == "__main__":
    main()
