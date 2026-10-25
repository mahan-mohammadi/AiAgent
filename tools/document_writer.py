

from langchain_core.tools import tool

from .document_reader import resolve_document


@tool
def write_document(
    filename: str,
    content: str,
    overwrite: bool = False,
) -> str:
    """Write content to a .txt or .md file inside documents/.
    Use for saving notes, summaries, reports, or drafts.
    Existing files are not overwritten unless overwrite=True.
    Provide a relative filename, such as 'summary.md'.
    """
    try:
        path = resolve_document(filename)

        if path.suffix.lower() not in {".txt", ".md"}:
            return "Error: writing is supported only for .txt and .md files."

        if len(content) > 50000:
            return "Error: content exceeds the 50,000-character limit."

        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists() and not overwrite:
            return (
                f"Error: {filename} already exists. "
                "Choose another filename or explicitly request replacement."
            )

        path.write_text(content, encoding="utf-8")

        relative = path.relative_to(resolve_document("."))

        return (
            f"Successfully wrote {len(content)} characters "
            f"to documents/{relative.as_posix()}."
        )

    except Exception as exc:
        return f"Document write failed: {type(exc).__name__}: {exc}"
