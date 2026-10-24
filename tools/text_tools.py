

from langchain_core.tools import tool


@tool
def count_words(text: str) -> dict:
    """Count words, characters, and lines in text."""
    return {
        "words": len(text.split()),
        "characters": len(text),
        "lines": len(text.splitlines()) if text else 0,
    }


@tool
def reverse_text(text: str) -> str:
    """Return the text in reverse order."""
    return text[::-1]


@tool
def find_text(text: str, query: str) -> dict:
    """Find case-insensitive occurrences of a phrase in text."""
    if not query:
        raise ValueError("The search query cannot be empty.")

    count = text.casefold().count(query.casefold())
    return {"found": count > 0, "occurrences": count}
