import streamlit as st
import requests

BASE_URL = "http://localhost:8001"

st.title("Wellcome to the RAG App")

st.set_page_config(page_title="RAG Chat", layout="centered")
st.title("RAG Chatbot")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat messages (like ChatGPT style)
for i, (role, message) in enumerate(st.session_state.chat_history):
    if role == "user":
        with st.chat_message("user"):
            st.markdown(message)
    else:
        with st.chat_message("assistant"):
            st.markdown(message)

# Input box at the bottom
if user_input := st.chat_input("Ask something..."):
    # Show user message instantly
    st.session_state.chat_history.append(("user", user_input))

    # Call backend
    try:
        response = requests.post(f"{BASE_URL}/ask", json={"question": user_input})
        if response.status_code == 200:
            answer = response.json().get("answer", "No answer returned.")
        else:
            answer = "Error: Failed to fetch response."
    except Exception as e:
        answer = f"Exception: {e}"

    # Show assistant message
    st.session_state.chat_history.append(("assistant", answer))
    
    st.rerun()
