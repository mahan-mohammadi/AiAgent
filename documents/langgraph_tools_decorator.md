# LangGraph Tools Decorator

## Overview

The `tools` decorator in LangGraph is a mechanism for registering and managing tools that can be used by agents during their execution. These tools allow agents to interact with external systems, perform computations, or access data dynamically.

## Basic Syntax

```python
from langgraph.tools import tool

@tool("get_weather", description="Returns the current weather in a city")
def get_weather(city: str) -> str:
    return f"The weather in {city} is sunny."
```

## Key Features

### 1. Tool Registration
The `@tool` decorator automatically registers the function as a tool that can be invoked by agents. It extracts metadata like name, description, and parameters from the function signature.

### 2. Parameter Validation
LangGraph validates input parameters against the function's type hints before execution, ensuring type safety and preventing runtime errors.

### 3. Tool Invocation
Tools can be called programmatically or through agent messages:

```python
# Programmatic invocation
result = get_weather("New York")

# Agent invocation via tools list
tools = [get_weather, calculate_sum]
agent.invoke({"tools": tools})
```

## Advanced Usage

### Custom Tool Metadata
You can provide additional metadata for better tool discovery:

```python
@tool("calculate_sum", description="Calculates the sum of two numbers")
def calculate_sum(a: int, b: int) -> int:
    return a + b
```

### Tool Groups
Tools can be organized into groups for better management:

```python
from langgraph.tools import tools

weather_tools = [get_weather]
math_tools = [calculate_sum]
```

## Best Practices

1. **Clear Descriptions**: Provide concise, accurate descriptions for tool discovery.
2. **Type Hints**: Use proper type hints for parameter validation.
3. **Error Handling**: Implement try-except blocks within tools to handle unexpected errors gracefully.
4. **Documentation**: Include docstrings with examples for better agent understanding.

## Example Workflow

```python
from langgraph.graph import StateGraph, END

# Define tools
@tool("get_user_data", description="Retrieves user information")
def get_user_data(user_id: int) -> dict:
    return {"id": user_id, "name": "John Doe"}

# Create graph
workflow = StateGraph(dict)
workflow.add_node("tools", lambda state: {**state, **get_user_data(state["user_id"])})
workflow.set_entry_point("tools")
workflow.add_edge("tools", END)

app = workflow.compile()
```

## Conclusion

LangGraph's tools decorator provides a flexible and type-safe way to extend agent capabilities with external functionality. By following best practices for tool design, developers can create robust, maintainable agent systems.
