import re
import unicodedata

TERM = re.compile(r"[^\W_]+(?:[-'][^\W_]+)*", re.UNICODE)


def simple_tokenizer(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = text.replace("’", "'")
    text = text.replace("‐", "-").replace("‑", "-").replace("–", "-").replace("—", "-")
    return TERM.findall(text)


TOKENIZERS = {"simple_tokenizer": simple_tokenizer}
