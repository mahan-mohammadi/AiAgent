from typing import Literal, NotRequired, TypedDict

from langgraph.graph import MessagesState


class PlanStep(TypedDict): #each step in the plan
    description: str 
    status: Literal["pending", "completed", "failed"] #forces a step to go to 3 states defined here
    result: NotRequired[str] # means an step doesnt need a result until execeution makes one


class AgentState(MessagesState):
    """State shared by the planner, executor, and evaluator."""

    plan: list[PlanStep]
    current_step_index: int
    results: list[str]
    evaluation: Literal["continue", "replan", "finish", "fail"]
    iteration: int
    final_answer: str
