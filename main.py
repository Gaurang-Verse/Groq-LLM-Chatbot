"""
Groq LLM Chatbot
A multi-model streaming chat app built with Streamlit and the Groq API.
"""

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from groq import Groq, GroqError

# Load environment variables from .env in the project root
PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")


class GroqAPI:
    """Handles API operations with Groq to generate chat responses."""

    def __init__(self, model_name: str):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "API key is missing. Please set the GROQ_API_KEY environment variable."
            )
        self.client = Groq(api_key=api_key)
        self.model_name = model_name

    def _response(self, messages: list[dict]):
        return self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0,
            max_tokens=4096,
            stream=True,
            stop=None,
        )

    def response_stream(self, messages: list[dict]):
        """Yield content deltas from a streaming completion."""
        for chunk in self._response(messages):
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class Message:
    """Manages chat messages within the Streamlit UI."""

    system_prompt = (
        "You are a professional AI assistant. "
        "Please generate responses in English to all user inputs."
    )

    def __init__(self):
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "system", "content": self.system_prompt}
            ]

    def add(self, role: str, content: str):
        st.session_state.messages.append({"role": role, "content": content})

    def display_chat_history(self):
        """Render every non-system message in session state."""
        for message in st.session_state.messages:
            if message["role"] == "system":
                continue
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    def display_stream(self, generator) -> str:
        """Render a streamed assistant response and return the full text."""
        with st.chat_message("assistant"):
            return st.write_stream(generator)


class ModelSelector:
    """Lets the user pick a model from Groq's currently supported list.

    Only models accessible on the free/developer API tier are listed.
    llama-3.1-8b-instant and llama-3.3-70b-versatile are Enterprise-only
    as of Sept 2026 — they 404 on a standard developer key.
    Check https://console.groq.com/docs/models before adding new ones.
    """

    def __init__(self):
        self.models = [
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
        ]

    def select(self) -> str:
        with st.sidebar:
            st.title("Groq LLM Chatbot")
            return st.selectbox("Select a model:", self.models)


def main():
    st.set_page_config(page_title="Groq LLM Chatbot", page_icon="💬")

    model_selector = ModelSelector()
    selected_model = model_selector.select()

    message = Message()
    message.display_chat_history()  # always show history, not just after new input

    user_input = st.chat_input("Enter a message...")
    if not user_input:
        return

    message.add("user", user_input)
    with st.chat_message("user"):
        st.markdown(user_input)

    try:
        llm = GroqAPI(selected_model)
        response = message.display_stream(
            llm.response_stream(st.session_state.messages)
        )
        message.add("assistant", response)
    except ValueError as e:
        st.error(str(e))
    except GroqError as e:
        st.error(f"Groq API error: {e}")


if __name__ == "__main__":
    main()