from pprint import pprint

from graph import executor

state = {
    "messages": [
        {
            "role": "user",
            "content": "Calculate 25 × 14, then explain the result.",
        }
    ],
    "plan": [
        {
            "description": "Calculate the product of 25 and 14",
            "status": "pending",
        },
        {
            "description": "Explain the result",
            "status": "pending",
        },
    ],
    "current_step_index": 0,
    "results": [],
    "evaluation": "continue",
    "iteration": 0,
}

result = executor(state)
print("Full response:")
pprint(result["messages"][0].model_dump())
for message in result["messages"]:
    print("Content:", message.content)
    print("Tool calls:")
    pprint(message.tool_calls)
