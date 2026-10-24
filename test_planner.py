from pprint import pprint

from graph import planner


result = planner({
    "messages": [
        {
            "role": "user",
            "content": "Calculate 25 × 14, then explain the result.",
        }
    ]
})

pprint(result)
