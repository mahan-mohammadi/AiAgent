

from graph import build_graph


graph = build_graph()


def main():
    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() in {"exit", "quit"}:
            break

        if not user_input:
            continue

        result = graph.invoke({
            "messages": [
                {"role": "user", "content": user_input}
            ]
        })

        for message in result["messages"]:
            if message.type == "ai" and message.tool_calls:
                for call in message.tool_calls:
                    print(f"\n[Tool call] {call['name']}")
                    print(f"[Arguments] {call['args']}")

            elif message.type == "tool":
                print(f"[Tool result] {message.content}")

        print("\nAgent:", result["messages"][-1].content)


if __name__ == "__main__":
    main()
