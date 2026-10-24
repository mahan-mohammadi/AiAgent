
import json
from langchain_core.messages import SystemMessage , HumanMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from config import (
    MODEL_NAME,
    MODEL_TEMPERATURE,
    SYSTEM_PROMPT,
    PLANNER_PROMPT,
    EXECUTOR_PROMPT,
)
from state import AgentState
from tools import (
    calculator,
    get_current_time,
    get_relative_date,
    time_until,
    count_words,
    reverse_text,
    find_text,
    convert_units,
)


TOOLS = [
    calculator,
    get_current_time,
    get_relative_date,
    time_until,
    count_words,
    reverse_text,
    find_text,
    convert_units,
]


llm = ChatOllama(
    model=MODEL_NAME,
    temperature=MODEL_TEMPERATURE,
)

llm_with_tools = llm.bind_tools(TOOLS)

def planner(state: AgentState):
    messages = [
        SystemMessage(content=PLANNER_PROMPT),
        *state["messages"],
    ]

    response = llm.invoke(messages) #we use llm not llm with tools since this is planning only
    content = response.content.strip()

    if content.startswith("```"):
        content = content.removeprefix("```json").removeprefix("```")
        content = content.removesuffix("```").strip()

    data = json.loads(content)
    plan = data["plan"]

    if not plan:
        raise ValueError("Planner returned an empty plan")

    for step in plan:
        if not isinstance(step.get("description"), str) or not step["description"].strip():
            raise ValueError("Each plan step needs a description")

        step["status"] = "pending"

    return {
        "plan": plan,
        "current_step_index": 0,
        "results": [],
        "evaluation": "continue",
        "iteration": 0,
    }

def executor(state: AgentState):
    plan = state["plan"]
    index = state["current_step_index"]

    if index >= len(plan):
        raise ValueError("No pending plan step to execute")

    current_step = plan[index]

    context = f"""
User request:
{state["messages"][0]["content"]}
Current step:
{current_step["description"]}

Previous results:
{state["results"]}
"""

    messages = [
        SystemMessage(content="Use a tool when needed. Do not calculate manually."),
        HumanMessage(content=context),
    ]

    response = llm_with_tools.invoke(messages)

    return {"messages": [response]}

def agent(state: AgentState):
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        *state["messages"],
    ]

    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode(TOOLS))

    builder.add_edge(START, "agent")

    builder.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "tools": "tools",
            END: END,
        },
    )

    builder.add_edge("tools", "agent")

    return builder.compile()
