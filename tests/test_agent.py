"""Automated tests for the agent. Run with:  uv run pytest

None of these need Ollama or the internet: the LLM is replaced by a
fake, and files/databases live in temporary folders.
"""

import importlib
import sys
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import memory
from plan_utils import fallback_plan, parse_plan

VALID_TOOLS = {"calculator", "remember_fact", "read_document"}


# --------------------------------------------------------------------
# Planner output parsing (plan_utils.parse_plan)
# --------------------------------------------------------------------

def test_parse_plan_accepts_valid_object():
    raw = '{"plan": [{"description": "Multiply", "status": "pending", "tool_name": "calculator"}]}'
    plan = parse_plan(raw, VALID_TOOLS)
    assert plan == [
        {"description": "Multiply", "status": "pending", "tool_name": "calculator"}
    ]


def test_parse_plan_accepts_bare_list():
    raw = '[{"description": "Say hi", "tool_name": null}]'
    assert parse_plan(raw, VALID_TOOLS)[0]["tool_name"] is None


def test_parse_plan_strips_code_fences():
    raw = '```json\n{"plan": [{"description": "Say hi", "tool_name": null}]}\n```'
    assert parse_plan(raw, VALID_TOOLS)[0]["description"] == "Say hi"


def test_parse_plan_finds_json_inside_prose():
    raw = 'Sure! Here you go: {"plan": [{"description": "Say hi", "tool_name": null}]} Hope it helps.'
    assert len(parse_plan(raw, VALID_TOOLS)) == 1


def test_parse_plan_ignores_think_blocks():
    raw = '<think>hmm {not json}</think>{"plan": [{"description": "Say hi", "tool_name": null}]}'
    assert len(parse_plan(raw, VALID_TOOLS)) == 1


def test_parse_plan_treats_null_string_as_no_tool():
    raw = '{"plan": [{"description": "Say hi", "tool_name": "null"}]}'
    assert parse_plan(raw, VALID_TOOLS)[0]["tool_name"] is None


def test_parse_plan_resets_status_and_drops_result():
    raw = '{"plan": [{"description": "x", "status": "completed", "result": "done", "tool_name": null}]}'
    step = parse_plan(raw, VALID_TOOLS)[0]
    assert step["status"] == "pending"
    assert "result" not in step


def test_parse_plan_rejects_invalid_json():
    with pytest.raises(ValueError, match="valid JSON"):
        parse_plan("I cannot make a plan", VALID_TOOLS)


def test_parse_plan_rejects_empty_plan():
    with pytest.raises(ValueError, match="empty"):
        parse_plan('{"plan": []}', VALID_TOOLS)


def test_parse_plan_rejects_unknown_tool():
    raw = '{"plan": [{"description": "x", "tool_name": "hack_the_planet"}]}'
    with pytest.raises(ValueError, match="unknown tool"):
        parse_plan(raw, VALID_TOOLS)


def test_parse_plan_rejects_non_string_tool_name():
    raw = '{"plan": [{"description": "x", "tool_name": ["calculator"]}]}'
    with pytest.raises(ValueError):
        parse_plan(raw, VALID_TOOLS)


def test_parse_plan_rejects_missing_description():
    with pytest.raises(ValueError, match="description"):
        parse_plan('{"plan": [{"tool_name": null}]}', VALID_TOOLS)


def test_parse_plan_rejects_non_object_step():
    with pytest.raises(ValueError, match="not a JSON object"):
        parse_plan('{"plan": ["just text"]}', VALID_TOOLS)


def test_parse_plan_rejects_non_text_content():
    with pytest.raises(ValueError):
        parse_plan([{"type": "text"}], VALID_TOOLS)


def test_fallback_plan_is_a_fresh_valid_plan():
    first, second = fallback_plan(), fallback_plan()
    assert first == second and first is not second
    assert first[0]["tool_name"] is None and first[0]["status"] == "pending"


# --------------------------------------------------------------------
# Saved memory (memory.py)
# --------------------------------------------------------------------

@pytest.fixture
def mem(tmp_path, monkeypatch):
    """memory.py pointed at an empty temporary database."""
    monkeypatch.setattr(memory, "DB_PATH", tmp_path / "memory.db")
    memory.init_memory()
    return memory


def test_save_memory_updates_existing_key(mem):
    mem.save_memory("favorite_color", "blue")
    mem.save_memory("favorite_color", "green")
    rows = mem.get_all_memories()
    assert len(rows) == 1 and rows[0]["value"] == "green"


def test_save_memory_rejects_empty_values(mem):
    assert "cannot be empty" in mem.save_memory("  ", "x")
    assert mem.get_all_memories() == []


def test_search_ignores_stopwords_and_short_words(mem):
    # "is" and "the" used to match almost everything via LIKE '%is%'.
    mem.save_memory("hometown", "this is Paris")
    assert mem.search_memory("What is the capital of France?") == []


def test_search_finds_relevant_memory(mem):
    mem.save_memory("favorite_color", "blue")
    mem.save_memory("pet_name", "Tom")
    found = mem.search_memory("What is my favorite color?")
    assert [m["key"] for m in found] == ["favorite_color"]


def test_search_ranks_more_matching_terms_first(mem):
    mem.save_memory("favorite_color", "blue")
    mem.save_memory("favorite_food", "pizza")
    keys = [m["key"] for m in mem.search_memory("favorite food")]
    assert keys == ["favorite_food", "favorite_color"]


def test_search_treats_wildcards_literally(mem):
    mem.save_memory("a_b", "one")
    mem.save_memory("axb", "two")
    assert [m["key"] for m in mem.search_memory("a_b")] == ["a_b"]


def test_asks_about_memory():
    assert memory.asks_about_memory("What do you remember about me?")
    assert memory.asks_about_memory("Show me my saved memories")
    assert not memory.asks_about_memory("remember that my cat is Tom")
    assert not memory.asks_about_memory("What is the capital of France?")


# --------------------------------------------------------------------
# Simple tools
# --------------------------------------------------------------------

def test_calculator_operations():
    from tools import calculator

    assert calculator.invoke({"numbers": [555, 10], "operation": "multiply"}) == 5550
    assert calculator.invoke({"numbers": [10, 4], "operation": "divide"}) == 2.5
    assert calculator.invoke({"numbers": [1, 2, 3], "operation": "average"}) == 2


def test_calculator_errors():
    from tools import calculator

    with pytest.raises(ValueError, match="divide by zero"):
        calculator.invoke({"numbers": [1, 0], "operation": "divide"})
    with pytest.raises(ValueError, match="Unknown operation"):
        calculator.invoke({"numbers": [1, 2], "operation": "modulo"})


def test_convert_units():
    from tools import convert_units

    km = convert_units.invoke(
        {"value": 1, "category": "length", "from_unit": "km", "to_unit": "m"}
    )
    assert "1000.0000 m" in km

    temp = convert_units.invoke(
        {"value": 212, "category": "temperature", "from_unit": "F", "to_unit": "C"}
    )
    assert "100.0000 C" in temp


def test_convert_units_rejects_bad_category():
    from tools import convert_units

    with pytest.raises(ValueError, match="Category"):
        convert_units.invoke(
            {"value": 1, "category": "volume", "from_unit": "l", "to_unit": "ml"}
        )


# --------------------------------------------------------------------
# Document tools
# --------------------------------------------------------------------

@pytest.fixture
def docs(tmp_path, monkeypatch):
    """Point the document tools at an empty temporary documents/ folder."""
    import tools.document_reader as reader

    folder = (tmp_path / "documents").resolve()
    folder.mkdir()
    monkeypatch.setattr(reader, "DOCUMENTS_DIR", folder)
    return folder


def test_resolve_document_blocks_path_escape(docs):
    from tools.document_reader import resolve_document

    with pytest.raises(ValueError, match="outside"):
        resolve_document("../secret.txt")
    with pytest.raises(ValueError, match="relative"):
        resolve_document("/etc/passwd")


def test_load_text_reads_markdown_and_docx(docs):
    from docx import Document
    from tools.document_reader import load_text

    (docs / "note.md").write_text("# Title\nbody", encoding="utf-8")
    assert "body" in load_text(docs / "note.md")

    word = Document()
    word.add_paragraph("Hello from Word")
    word.save(str(docs / "file.docx"))
    assert "Hello from Word" in load_text(docs / "file.docx")


def test_load_text_rejects_unsupported_type(docs):
    from tools.document_reader import load_text

    with pytest.raises(ValueError, match="Unsupported"):
        load_text(docs / "program.exe")


def test_load_text_does_not_truncate(docs):
    from tools.document_reader import load_text

    (docs / "long.txt").write_text("a" * 50_000, encoding="utf-8")
    assert len(load_text(docs / "long.txt")) == 50_000


def test_read_document_success_and_truncation(docs):
    from tools import read_document

    (docs / "short.txt").write_text("hello", encoding="utf-8")
    assert "hello" in read_document.invoke({"filename": "short.txt"})

    (docs / "long.txt").write_text("a" * 20_000, encoding="utf-8")
    out = read_document.invoke({"filename": "long.txt"})
    assert "[Content truncated.]" in out and len(out) < 12_500


def test_read_document_error_messages(docs):
    from tools import read_document

    assert "not found" in read_document.invoke({"filename": "missing.txt"})
    (docs / "tool.exe").write_text("x")
    assert "supported formats" in read_document.invoke({"filename": "tool.exe"})
    (docs / "empty.txt").write_text("   ")
    assert "no extractable text" in read_document.invoke({"filename": "empty.txt"})
    assert "read failed" in read_document.invoke({"filename": "../escape.txt"})


def test_write_document_creates_file_with_correct_path_message(docs):
    from tools import write_document

    out = write_document.invoke({"filename": "summary.md", "content": "hi"})
    assert (docs / "summary.md").read_text(encoding="utf-8") == "hi"
    assert "documents/summary.md" in out
    assert "documents/documents" not in out


def test_write_document_does_not_overwrite_by_default(docs):
    from tools import write_document

    write_document.invoke({"filename": "a.md", "content": "one"})
    out = write_document.invoke({"filename": "a.md", "content": "two"})
    assert "already exists" in out
    assert (docs / "a.md").read_text(encoding="utf-8") == "one"

    write_document.invoke({"filename": "a.md", "content": "two", "overwrite": True})
    assert (docs / "a.md").read_text(encoding="utf-8") == "two"


def test_write_document_blocks_bad_types_and_escapes(docs):
    from tools import write_document

    assert "only for .txt and .md" in write_document.invoke(
        {"filename": "script.py", "content": "print(1)"}
    )
    assert "failed" in write_document.invoke(
        {"filename": "../evil.md", "content": "x"}
    )
    assert not (docs.parent / "evil.md").exists()


# --------------------------------------------------------------------
# Graph logic (routing, evaluator, planner) with a fake LLM
# --------------------------------------------------------------------

@pytest.fixture(scope="module")
def graph_module(tmp_path_factory):
    """Import graph.py once, from a temp cwd so its checkpoints.sqlite
    is created there and not in the project folder."""
    patch = pytest.MonkeyPatch()
    patch.chdir(tmp_path_factory.mktemp("graph"))
    sys.modules.pop("graph", None)
    module = importlib.import_module("graph")
    yield module
    patch.undo()
    sys.modules.pop("graph", None)


class FakeLLM:
    """Returns canned replies in order and records what it was sent."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []

    def invoke(self, messages):
        self.seen.append(messages)
        return SimpleNamespace(content=self.replies.pop(0))


def step(tool_name=None, description="do something"):
    return {"description": description, "status": "pending", "tool_name": tool_name}


def ai_call(name, **args):
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": "call_1", "type": "tool_call"}],
    )


def eval_state(plan, message, **extra):
    state = {
        "messages": [message],
        "plan": plan,
        "current_step_index": 0,
        "results": [],
        "iteration": 0,
        "retry_count": 0,
    }
    state.update(extra)
    return state


# ---- route_evaluator / route_executor ----

def test_route_evaluator(graph_module):
    route = graph_module.route_evaluator
    assert route({"evaluation": "continue"}) == "executor"
    assert route({"evaluation": "retry"}) == "executor"
    assert route({"evaluation": "finish"}) == "finalize"
    assert route({"evaluation": "fail"}) == graph_module.END


def test_route_executor_accepts_only_the_expected_tool(graph_module):
    route = graph_module.route_executor
    plan = [step("calculator")]

    def state_for(message, plan=plan):
        return {"messages": [message], "plan": plan, "current_step_index": 0}

    assert route(state_for(ai_call("calculator", numbers=[1], operation="add"))) == "tools"
    assert route(state_for(ai_call("web_search", query="x"))) == "evaluator"
    assert route(state_for(AIMessage(content="just text"))) == "evaluator"
    # a tool call on a step that should use no tool is rejected
    assert route(state_for(ai_call("calculator"), plan=[step(None)])) == "evaluator"


# ---- evaluator ----

def test_evaluator_tool_success_moves_to_next_step(graph_module):
    plan = [step("calculator"), step(None)]
    state = eval_state(plan, ToolMessage(content="350", tool_call_id="call_1"))
    out = graph_module.evaluator(state)
    assert out["evaluation"] == "continue"
    assert out["current_step_index"] == 1
    assert out["results"] == ["350"]
    assert out["plan"][0]["status"] == "completed"


def test_evaluator_finishes_after_last_step(graph_module):
    state = eval_state([step("calculator")], ToolMessage(content="350", tool_call_id="c"))
    assert graph_module.evaluator(state)["evaluation"] == "finish"


def test_evaluator_retries_when_model_skips_the_tool(graph_module):
    state = eval_state([step("calculator")], AIMessage(content="It is 350."))
    out = graph_module.evaluator(state)
    assert out["evaluation"] == "retry" and out["retry_count"] == 1
    assert "calculator" in out["last_error"]


def test_evaluator_retries_on_tool_error_text(graph_module):
    err = ToolMessage(content="Error invoking tool 'calculator' with kwargs {}", tool_call_id="c")
    out = graph_module.evaluator(eval_state([step("calculator")], err))
    assert out["evaluation"] == "retry"


def test_evaluator_fails_after_max_retries(graph_module):
    state = eval_state(
        [step("calculator")],
        AIMessage(content="no tool"),
        retry_count=graph_module.MAX_RETRIES,
    )
    out = graph_module.evaluator(state)
    assert out["evaluation"] == "fail"
    assert out["plan"][0]["status"] == "failed"


def test_evaluator_no_tool_step_accepts_text(graph_module):
    out = graph_module.evaluator(eval_state([step(None)], AIMessage(content="Hello!")))
    assert out["evaluation"] == "finish" and out["results"] == ["Hello!"]


def test_evaluator_no_tool_step_rejects_tool_call(graph_module):
    state = eval_state([step(None)], ai_call("calculator"))
    assert graph_module.evaluator(state)["evaluation"] == "retry"


def test_evaluator_stops_at_step_limit(graph_module):
    plan = [step(None), step(None), step(None)]
    state = eval_state(
        plan, AIMessage(content="ok"), iteration=graph_module.MAX_STEPS - 1
    )
    out = graph_module.evaluator(state)
    assert out["evaluation"] == "finish"
    assert any("maximum step limit" in r for r in out["results"])


# ---- planner ----

GOOD_PLAN = '{"plan": [{"description": "Multiply 25 by 14", "status": "pending", "tool_name": "calculator"}]}'


def planner_state(text):
    return {"messages": [HumanMessage(content=text)], "memories": []}


def test_planner_happy_path(graph_module, monkeypatch):
    fake = FakeLLM([GOOD_PLAN])
    monkeypatch.setattr(graph_module, "llm", fake)
    out = graph_module.planner(planner_state("What is 25 times 14?"))
    assert out["plan"][0]["tool_name"] == "calculator"
    assert out["current_step_index"] == 0 and len(fake.seen) == 1


def test_planner_retries_once_after_bad_json(graph_module, monkeypatch):
    fake = FakeLLM(["Sure, here is the plan!", GOOD_PLAN])
    monkeypatch.setattr(graph_module, "llm", fake)
    out = graph_module.planner(planner_state("What is 25 times 14?"))
    assert out["plan"][0]["tool_name"] == "calculator"
    assert len(fake.seen) == 2
    assert "rejected" in fake.seen[1][-1].content  # error fed back to model


def test_planner_falls_back_instead_of_crashing(graph_module, monkeypatch):
    fake = FakeLLM(["not json", "still not json"])
    monkeypatch.setattr(graph_module, "llm", fake)
    out = graph_module.planner(planner_state("What is 25 times 14?"))
    assert len(out["plan"]) == 1
    assert out["plan"][0]["tool_name"] is None
    assert out["plan"][0]["status"] == "pending"


def test_planner_forces_remember_fact_for_save_requests(graph_module, monkeypatch):
    plan = '{"plan": [{"description": "Reply", "tool_name": null}]}'
    monkeypatch.setattr(graph_module, "llm", FakeLLM([plan]))
    out = graph_module.planner(planner_state("remember that my cat is Tom"))
    assert out["plan"][0]["tool_name"] == "remember_fact"


def test_planner_never_saves_on_recall_questions(graph_module, monkeypatch):
    plan = '{"plan": [{"description": "Look up", "tool_name": "remember_fact"}]}'
    monkeypatch.setattr(graph_module, "llm", FakeLLM([plan]))
    out = graph_module.planner(planner_state("What is my cat's name?"))
    assert out["plan"][0]["tool_name"] is None


# ---- retrieve_memories ----

def test_retrieve_memories_has_no_fallback_for_unrelated_questions(graph_module, monkeypatch):
    monkeypatch.setattr(graph_module, "search_memory", lambda q: [])
    monkeypatch.setattr(graph_module, "get_all_memories", lambda: [{"id": 1}])
    state = {"messages": [HumanMessage(content="What is the capital of France?")]}
    assert graph_module.retrieve_memories(state) == {"memories": []}


def test_retrieve_memories_lists_latest_when_user_asks_what_is_saved(graph_module, monkeypatch):
    monkeypatch.setattr(graph_module, "search_memory", lambda q: [])
    monkeypatch.setattr(
        graph_module, "get_all_memories", lambda: [{"id": i} for i in range(1, 8)]
    )
    state = {"messages": [HumanMessage(content="What do you remember about me?")]}
    ids = [m["id"] for m in graph_module.retrieve_memories(state)["memories"]]
    assert ids == [7, 6, 5, 4, 3]  # newest first, max 5


# ---- executor prompt (build_step_context) ----

def context_state(**extra):
    state = {
        "messages": [HumanMessage(content="Read test.txt, summarize it, count the words")],
        "results": [],
        "memories": [],
    }
    state.update(extra)
    return state


def test_step_context_includes_request_and_earlier_results(graph_module):
    state = context_state(results=["DOC TEXT about The Kinks", "One sentence summary."])
    text = graph_module.build_step_context(
        state, step("count_words", "Count the words in the summary"), "count_words"
    )
    assert "Read test.txt, summarize it" in text      # original request
    assert "One sentence summary." in text            # earlier result
    assert "Call the tool `count_words` exactly once" in text
    assert "Do not discuss previous results" not in text


def test_step_context_for_no_tool_step_forbids_tools_and_rambling(graph_module):
    text = graph_module.build_step_context(
        context_state(), step(None, "Summarize in a single sentence"), None
    )
    assert "Do not call any tool" in text
    assert "Required tool" not in text and "Call the tool" not in text


def test_step_context_feeds_back_the_last_error(graph_module):
    state = context_state(last_error="Expected an actual call to count_words")
    text = graph_module.build_step_context(state, step("count_words"), "count_words")
    assert "previous attempt failed" in text
    assert "Expected an actual call to count_words" in text


def test_step_context_omits_empty_sections(graph_module):
    text = graph_module.build_step_context(context_state(), step(None), None)
    assert "Results of earlier steps" not in text
    assert "Saved memories" not in text
    assert "previous attempt failed" not in text
