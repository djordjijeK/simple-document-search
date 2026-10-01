from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[2] / "data"
DATASET_NAME = "scifact"
DATASET_PATH = DATA_PATH / DATASET_NAME

URL = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{DATASET_NAME}.zip"
ARCHIVE_FILENAME = f"{DATASET_NAME}.zip"
DOWNLOAD_FILENAME = f"{ARCHIVE_FILENAME}.download"

CORPUS_FILENAME = "corpus.jsonl"
QUERIES_FILENAME = "queries.jsonl"
QRELS_FILENAMES = ("qrels/train.tsv", "qrels/test.tsv")

ARCHIVE_FILES = tuple(
    f"{DATASET_NAME}/{filename}"
    for filename in (CORPUS_FILENAME, QUERIES_FILENAME, *QRELS_FILENAMES)
)

INDEX_PATH = DATA_PATH / f"{DATASET_NAME}-index.json"
