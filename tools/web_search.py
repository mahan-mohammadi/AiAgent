from ddgs import DDGS
from langchain_core.tools import tool


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search DuckDuckGo for web pages relevant to a query.

    Use for current information, recent events, facts that need
    verification, and questions requiring online research.
    """
    if not query.strip():
        return "Search failed: query cannot be empty."

    max_results = max(1, min(max_results, 8))

    try:
        results = DDGS(timeout=10).text(
            query,
            max_results=max_results,
            backend="duckduckgo",
        )

        results = list(results)

        if not results:
            return f"No search results found for: {query}"

        formatted = []

        for i, result in enumerate(results, start=1):
            formatted.append(
                f"{i}. {result.get('title', 'Untitled')}\n"
                f"URL: {result.get('href', '')}\n"
                f"Snippet: {result.get('body', '')}"
            )

        return "\n\n".join(formatted)

    except Exception as exc:
        return f"Search failed: {type(exc).__name__}: {exc}"
