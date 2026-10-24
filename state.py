from typing import Literal, NotRequired, TypedDict

from langgraph.graph import MessagesState


class PlanStep(TypedDict): #each step in the plan
    description: str 
    status: Literal["pending", "completed", "failed"] #forces a step to go to 3 states defined here

    tool_name: NotRequired[str | None]
    result: NotRequired[str] # means an step doesnt need a result until execeution makes one

class AgentState(MessagesState):
    plan: list[PlanStep]
    current_step_index: int
    results: list[str]
    evaluation: Literal["continue", "replan", "finish", "fail", "retry"]
    iteration: int
    final_answer: str
    retry_count: int
    last_error: str
    memories: list[dict]
