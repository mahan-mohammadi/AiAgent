
from langchain_core.tools import tool
from memory import save_memory


@tool
def remember_fact(key: str, value: str) -> str:
    """Save a user-provided fact or preference for future conversations.

    Use this when the user explicitly asks you to remember or save
    something. Use a short, descriptive key and store the information
    in value.
    """
    return save_memory(key, value)
