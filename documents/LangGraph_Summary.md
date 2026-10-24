# LangGraph Summary

## Overview
LangGraph is an open-source framework from LangChain designed to build and manage AI agent workflows using graph-based structures. It allows developers to define workflows as nodes and edges, making complex agent interactions more structured, scalable, and easier to control.

## Key Features
- **Durable Execution**: Ensures reliable execution of agent workflows even in the presence of failures or interruptions.
- **Streaming**: Supports real-time streaming of outputs during agent execution.
- **Human-in-the-loop**: Enables human intervention at specific points in the workflow.
- **Deterministic and LLM-driven Steps**: Can mix deterministic steps with LLM-driven agentic steps in a single graph.
- **State Management**: Manages memory and state across multiple agents and complex workflows.
- **Tool Integration**: Supports integration of external tools for enhanced agent capabilities.

## Differences from Other Frameworks
While other agentic frameworks can handle simple, generic tasks, LangGraph provides a more expressive framework to handle companies' unique tasks without restricting users to a single black-box cognitive architecture. It is particularly suited for complex, bespoke tasks that require structured control over agent interactions.

## Use Cases
- Building stateful, multi-agent AI agents
- Managing memory and cycles in LLM workflows
- Creating custom agent orchestration systems for specific business needs