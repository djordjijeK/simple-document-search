from collections.abc import Sequence
from math import log2
from statistics import mean

from document_search.index import InvertedIndex
from document_search.retriever import BM25Retriever, TfIdfRetriever
from document_search.scifact_data_loader import load_scifact_data, load_scifact_qrels
from document_search.tokenizer import simple_tokenizer


def ranking_metrics(
    ranked_ids: Sequence[str], relevant_ids: set[str], k: int = 10
) -> dict[str, float]:
    if k < 1:
        raise ValueError("k must be positive")

    if not relevant_ids:
        raise ValueError("A query must have at least one relevant document")

    top_ids = ranked_ids[:k]

    hits = [document_id in relevant_ids for document_id in top_ids]

    recall_at_k = sum(hits) / len(relevant_ids)

    reciprocal_rank_at_k = next(
        (1 / rank for rank, hit in enumerate(hits, start=1) if hit),
        0.0,
    )

    dcg = sum(1 / log2(1 + rank) for rank, hit in enumerate(hits, start=1) if hit)

    ideal_dcg = sum(
        1 / log2(rank + 1) for rank in range(1, min(len(relevant_ids), k) + 1)
    )

    return {
        f"recall@{k}": recall_at_k,
        f"rr@{k}": reciprocal_rank_at_k,
        f"nDCG@{k}": dcg / ideal_dcg,
    }


def evaluate_retrievers(split: str = "train", k: int = 10) -> dict:
    if k < 1:
        raise ValueError("k must be positive")

    documents, queries = load_scifact_data()
    qrels = load_scifact_qrels(split)
    if not qrels:
        raise ValueError("No evaluation queries found")

    document_texts = {
        document_id: document.content for document_id, document in documents.items()
    }
    index = InvertedIndex.build(document_texts, simple_tokenizer)

    report = {"split": split, "k": k, "query_count": len(qrels), "retrievers": {}}
    retrievers = {
        "TF-IDF": TfIdfRetriever(index),
        "BM25": BM25Retriever(index, k1=1.2, b=0.75),
    }

    metric_keys = {
        f"Recall@{k}": f"recall@{k}",
        f"MRR@{k}": f"rr@{k}",
        f"nDCG@{k}": f"nDCG@{k}",
    }

    for name, retriever in retrievers.items():
        per_query = {}
        for query_id, relevant_ids in sorted(qrels.items()):
            ranking = retriever.search(queries[query_id])[:k]
            metrics = ranking_metrics(
                [document_id for document_id, _ in ranking], relevant_ids, k
            )
            per_query[query_id] = {"ranking": ranking, "metrics": metrics}

        averages = {
            label: mean(row["metrics"][key] for row in per_query.values())
            for label, key in metric_keys.items()
        }

        report["retrievers"][name] = {"mean": averages, "per_query": per_query}

    return report


if __name__ == "__main__":
    for split in ("train", "test"):
        report = evaluate_retrievers(split=split, k=10)
        print(f"\n{report['split']}: {report['query_count']} queries")

        labels = next(iter(report["retrievers"].values()))["mean"]
        print(
            f"{'Retriever':<12} "
            + " ".join(f"{label:>10}" for label in labels)
        )

        for name, result in report["retrievers"].items():
            values = " ".join(
                f"{value:10.4f}" for value in result["mean"].values()
            )
            print(f"{name:<12} {values}")
