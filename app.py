import uuid


import streamlit as st
from graph import build_graph
from chat_storage import (
    initialize_db,
    create_conversation,
    save_message,
    load_messages,
    list_conversations,
)


initialize_db()

st.set_page_config(
    page_title="LangGraph Agent",
    page_icon="🤖",
    layout="wide",
)

st.title("LangGraph Agent")
st.caption("Local AI agent powered by LangGraph and Ollama")

# Keep the graph and UI session alive across Streamlit reruns.
@st.cache_resource
def get_graph():
    return build_graph()


graph = get_graph()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_created" not in st.session_state:
    st.session_state.conversation_created = False
if "trace" not in st.session_state:
    st.session_state.trace = None


# Sidebar
with st.sidebar:
    st.header("Controls")
    if st.button("＋ New conversation", use_container_width=True):
        st.session_state.thread_id = None
        st.session_state.messages = []
        st.session_state.trace = None
        st.rerun()
    st.divider()
    st.caption("Model: local Ollama")
    st.caption("Memory: existing agent memory system")
    st.caption("Conversation history: current thread")
    
    conversations = list_conversations()
    if conversations:
        st.subheader("Previous conversations")

        for conversation in conversations:
            title = conversation["title"]
            thread_id = conversation["thread_id"]

            if st.button(title, key=f"chat_{thread_id}"):
                st.session_state.thread_id = thread_id
                st.session_state.messages = load_messages(thread_id)
                st.session_state.trace = None
                st.rerun()
# Display conversation history.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Handle new messages.
prompt = st.chat_input("Ask your agent something...")

if prompt:
    # Save and display the user's message immediately.
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )
    

    save_message(st.session_state.thread_id , "user" , prompt)

    with st.chat_message("user"):
        st.markdown(prompt)

    # Run the agent and save the answer.
    with st.chat_message("assistant"):
        try:
            with st.spinner("Agent is working..."):
                result = graph.invoke(
                    {"messages": [{"role": "user", "content": prompt}]},
                    config={
                        "recursion_limit": 25,
                        "configurable": {
                            "thread_id": st.session_state.thread_id
                        },
                    },
                )

            if result.get("evaluation") == "fail":
                answer = (
                    "I couldn't complete the request because "
                    "a required step failed."
                )
            else:
                answer = result["messages"][-1].content

            st.markdown(answer)

            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )


            save_message(
                st.session_state.thread_id,
                "assistant",
                answer,
            )

            st.session_state.trace = {
                "plan": result.get("plan", []),
                "results": result.get("results", []),
                "evaluation": result.get("evaluation"),
                "iteration": result.get("iteration"),
                "last_error": result.get("last_error"),
            }

        except Exception as exc:
            error_message = f"Agent error: {exc}"
            st.error(error_message)
            st.session_state.messages.append(
                {"role": "assistant", "content": f"Agent error: {exc}"}
            )
            save_message(
                    st.session_state.thread_id,
                    "assistant",
                    error_message,
                )

    st.rerun()

# Agent execution trace
if st.session_state.trace:
    with st.expander("Agent activity", expanded=False):
        trace = st.session_state.trace

        st.subheader("Plan")
        for i, step in enumerate(trace["plan"], start=1):
            status = step.get("status", "pending")
            st.markdown(
                f"**{i}. {step.get('description', 'Unnamed step')}** "
                f"— `{status}`"
            )

            if step.get("tool_name"):
                st.caption(f"Tool: {step['tool_name']}")

        st.subheader("Step results")
        for i, result in enumerate(trace["results"], start=1):
            st.markdown(f"**Step {i}**")
            st.write(result)

        st.subheader("Evaluation")
        st.write(trace["evaluation"])
        st.write(f"Iteration: {trace['iteration']}")

        if trace["last_error"]:
            st.warning(trace["last_error"])
