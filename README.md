
# LangGraph Local AI Agent

A local AI agent built with **LangGraph**, **LangChain**, and **Ollama**
for a university assignment. It demonstrates planning, tool use,
multi-step execution, evaluation/recovery, memory, and a Streamlit chat
interface. It uses a locally hosted language model rather than requiring
a paid model API.

## Features

-   Local model inference through `ChatOllama`.
-   Graph workflow with memory retrieval, planner, executor, tool node,
    evaluator, and finalization.
-   Multi-step plans with step status and collected results.
-   Tool use for calculations, date/time, text processing, unit
    conversion, memory, web search, Python execution, data analysis, and
    document operations. Check the current `TOOLS` list for the exact
    enabled tools.
-   SQLite storage for conversations and visible chat messages.
-   Streamlit interface with previous conversations and an agent
    activity trace.
-   A command-line interface through `main.py`.

## Architecture

``` text
User message
    |
    v
Retrieve memories
    |
    v
Planner
    |
    v
Executor <--------------------+
    |                         |
    | tool required            | continue / retry / replan
    v                         |
ToolNode                      |
    |                         |
    v                         |
Evaluator --------------------+
    |
    | finish / fail
    v
Finalize
    |
    v
Response
```

The exact routing is defined in `graph.py`.

## Project structure

  -----------------------------------------------------------------------
  File                                Purpose
  ----------------------------------- -----------------------------------
  `app.py`                            Streamlit UI, chat display,
                                      conversation selection, and
                                      activity trace.

  `main.py`                           Command-line interface.

  `graph.py`                          LLM configuration, tool
                                      registration, graph nodes, routing,
                                      and compilation.

  `state.py`                          Shared LangGraph state and
                                      plan-step types.

  `tools.py`                          Tool definitions or imports; check
                                      `TOOLS` in `graph.py` for the
                                      enabled list.

  `memory.py`                         Saved-memory retrieval and storage
                                      helpers.

  `chat_storage.py`                   SQLite persistence for UI
                                      conversations and messages.

  `chat_history.sqlite`               Database created by
                                      `chat_storage.py`.

  `pyproject.toml`                    Project metadata and dependencies
                                      managed by `uv`.

  `uv.lock`                           Locked dependency versions, if
                                      present.
  -----------------------------------------------------------------------

Some tool implementations may live in separate modules if the project
has been reorganized.

## How it works

1.  **Retrieve memories:** find saved facts relevant to the current
    request.
2.  **Plan:** the planner uses the request and available context to
    create steps.
3.  **Execute:** the executor works on the current step and requests a
    tool when needed.
4.  **Use tools:** LangGraph's `ToolNode` runs the selected tool.
5.  **Evaluate:** the evaluator decides whether to continue, retry,
    replan, finish, or fail.
6.  **Finalize:** prepare the response for the user. The UI stores the
    visible exchange and can show the execution trace.

### State

`state.py` defines the shared state, based on LangGraph's
`MessagesState`. Fields used by the project include `messages`, `plan`,
`current_step_index`, `results`, `evaluation`, `iteration`,
`final_answer`, `retry_count`, `last_error`, and `memories`. Check
`state.py` for the authoritative types and optional fields.

### Memory versus conversation history

-   **Conversation history** is the messages exchanged in a particular
    chat.
-   **Saved memory** is information stored separately and retrieved for
    later requests.

They are separate systems and should not be treated as interchangeable.

## Requirements

-   Python version compatible with the dependencies in `pyproject.toml`
    (development has used Python 3.14).
-   [`uv`](https://docs.astral.sh/uv/) for dependency management.
-   [Ollama](https://ollama.com/) installed and running.
-   The model configured by `MODEL_NAME` in `graph.py` downloaded in
    Ollama.
-   Any system dependencies required by optional tools.

## Setup

### 1. Open the project

``` bash
cd /home/mahan/langgraph
```

### 2. Install Python dependencies

``` bash
uv sync
```

If Streamlit is not in the project dependencies yet:

``` bash
uv add streamlit
```

### 3. Check Ollama

``` bash
ollama list
```

If the configured model is missing, pull the matching model. For
example, if `MODEL_NAME` is `qwen3.5:4b`:

``` bash
ollama pull qwen3.5:4b
```

Use the exact model configured in your code; the example is not
mandatory.

### 4. Verify configuration

In `graph.py`, check the model name and settings. The project has used a
configuration similar to:

``` python
llm = ChatOllama(
    model=MODEL_NAME,
    temperature=MODEL_TEMPERATURE,
    reasoning=False,
)
```

Keep `MODEL_NAME` consistent with an installed Ollama model. Larger
context sizes can increase latency and memory use, so change them
deliberately.

## Run the application

### Streamlit UI

``` bash
uv run streamlit run app.py
```

Open the local URL printed by Streamlit. The UI supports new
conversations, selecting previous conversations, sending messages, and
inspecting the plan/results/evaluation in the **Agent activity**
expander.

### Command-line UI

``` bash
uv run python main.py
```

## Conversation persistence

The UI uses SQLite through `chat_storage.py`. The database has a
`conversations` table (thread ID, title, creation time) and a `messages`
table (message ID, thread ID, role, content, timestamp). Conversation
titles start as `New conversation` and are intended to update after the
first user message.

Each chat has a `thread_id`, which is passed to LangGraph as
`configurable.thread_id`.

**Important:** UI history and LangGraph checkpoints are separate. The
current graph configuration uses `InMemorySaver`, which loses checkpoint
state when the process exits. SQLite can preserve the messages shown in
the UI, but it does not automatically restore the graph checkpoint. Use
a persistent LangGraph checkpointer if the agent must resume internal
state after a restart.

Create a database conversation only when needed (ideally when the first
message is sent), rather than every time Streamlit starts. This avoids
empty `New conversation` entries. When selecting an existing
conversation, restore its thread ID and load its messages.

### Inspect the database

From the project directory:

``` bash
uv run python -c "from chat_storage import get_connection; c=get_connection(); print('Conversations:', c.execute('SELECT COUNT(*) FROM conversations').fetchone()[0]); print('Messages:', c.execute('SELECT COUNT(*) FROM messages').fetchone()[0]); c.close()"
```

List conversation titles:

``` bash
uv run python -c "from chat_storage import list_conversations; print(*[(c['thread_id'], repr(c['title'])) for c in list_conversations()], sep='\\n')"
```

Back up `chat_history.sqlite` before manually deleting records. Empty
entries usually mean conversations were created but no messages were
saved.

## Adding a tool

1.  Implement a function with a clear name, description, argument
    schema, and return value.
2.  Import it where the tool list is assembled.
3.  Add it to the `TOOLS` list used by the graph.
4.  Test it independently.
5.  Test a request that should naturally select it.
6.  Verify the executor/evaluator behavior when it succeeds or raises an
    error.

Local models can be sensitive to complex tool schemas, so test with the
model used by the project.

## Troubleshooting

### Agent is slow

-   Check which model is configured and loaded by Ollama.
-   Avoid increasing context size without a reason.
-   Keep prompts and tool descriptions concise.
-   Avoid unnecessary extra LLM calls.
-   Check GPU and CPU utilization.

### Many `New conversation` entries

They are likely empty rows in the SQLite database or entries whose
titles were not updated. Inspect the database using the commands above.
Fix the creation logic first; back up the database before deleting
records.

### UI messages remain, but agent context is lost after restart

This is expected with `InMemorySaver`. UI messages are persisted
separately in SQLite, but graph checkpoints are in memory only.

### Tool calls fail

-   Confirm the tool is registered in `TOOLS`.
-   Match tool names and argument names to the schema.
-   Try a simpler request.
-   Inspect the tool error and activity trace.
-   Confirm required files, programs, network access, and packages are
    available.

### Ollama cannot find the model

``` bash
ollama list
```

Check that `MODEL_NAME` exactly matches an installed model tag.

## Suggested university demonstration

Use a small set of repeatable tests:

1.  A simple calculation or date request.
2.  A multi-step request that needs more than one tool.
3.  Saving a non-sensitive fact and recalling it later.
4.  A request that triggers a tool error, to demonstrate recovery
    behavior.
5.  A follow-up question in the same conversation.
6.  The Agent activity panel, showing the plan, results, and evaluation.

Use only tools actually enabled in the current code. Record both
successes and failures, and evaluate task success, number of steps/tool
calls, failure recovery, and answer quality.

## Future improvements

-   Replace `InMemorySaver` with a persistent LangGraph checkpointer.
-   Add unit tests for tools and graph routing.
-   Test retry limits and error handling.
-   Improve memory relevance.
-   Add conversation rename/delete features with safe database handling.
-   Improve UI error messages and loading feedback.
-   Document the final model, tool list, and measured evaluation
    results.

## Security and privacy

-   Review tools that run Python, read/write files, or access the web
    before using them on important data.
-   Do not commit secrets or private information.
-   `chat_history.sqlite` may contain user messages; review it before
    adding it to a public repository.
-   Consider adding the SQLite database, Python caches, and local
    virtual environments to `.gitignore`.

## License

Choose and add a license before distributing the project. Do not claim a
license until one has been selected.
