from collections.abc import Sequence
from pathlib import Path
from shutil import copyfileobj
from urllib.request import urlopen
from zipfile import ZipFile

from document_search.scifact_config import (
    ARCHIVE_FILENAME,
    ARCHIVE_FILES,
    DATA_PATH,
    DATASET_NAME,
    DOWNLOAD_FILENAME,
    URL,
)


def download_scifact_data() -> Path:
    path = DATA_PATH
    path.mkdir(parents=True, exist_ok=True)

    archive = path / ARCHIVE_FILENAME

    if archive.is_file():
        extract_files(archive, path)
        return path / DATASET_NAME

    temporary = path / DOWNLOAD_FILENAME

    try:
        with urlopen(URL, timeout=60) as response:
            with temporary.open("wb") as local_file:
                copyfileobj(response, local_file)

        extract_files(temporary, path)
        temporary.replace(archive)
    finally:
        temporary.unlink(missing_ok=True)

    return path / DATASET_NAME


def extract_files(
    archive: Path, destination: Path, files: Sequence[str] = ARCHIVE_FILES
) -> None:
    with ZipFile(archive) as zip_file:
        available_files = set(zip_file.namelist())
        missing_files = [name for name in files if name not in available_files]

        if missing_files:
            raise ValueError(
                f"Required files missing from SciFact ZIP: {missing_files}"
            )

        zip_file.extractall(path=destination, members=files)


if __name__ == "__main__":
    download_scifact_data()
