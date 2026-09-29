# Simple Document Ranking

A small, inspectable TF-IDF document ranker. The CLI currently searches a
five-document example collection; BM25 and SciFact evaluation are planned.

Run a query from the project directory:

```sh
uv run simple-document-ranking "gene disease"
```

Save the example collection's index, then search from the saved index:

```sh
uv run simple-document-ranking "gene disease" --save-index index.json
uv run simple-document-ranking "gene disease" --load-index index.json
```

The index stores term frequencies, document lengths, document IDs, and the
tokenizer name. It does not store the original document text.
