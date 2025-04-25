from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import streamlit as st
from dotenv import load_dotenv
import os

# --- Load .env variables ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("Google API Key not found. Please set it in your .env file.")
    st.stop()

# --- Streamlit Page Setup ---
st.set_page_config(page_title="Reframe IT", page_icon="🌿")
st.title("Reframe IT")
st.subheader("Turn your Negative thoughts into positive")
st.caption("Amity Center of Happiness by Rekhi Foundation ")

# --- Model Setup ---
try:
    model = ChatGoogleGenerativeAI(
        model='gemini-1.5-flash',
        google_api_key=GOOGLE_API_KEY,
        temperature=0.7,
    )
except Exception as e:
    st.error(f"Model init failed: {e}")
    st.stop()

# --- Prompt Template Setup ---
cbt_few_shot_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a supportive therapist who uses CBT to help reframe negative thoughts."),

    # 🧠 Example 1
    ("human", 'Negative Thought: "I always mess things up."'),
    ("ai", "It sounds like you're being really hard on yourself. Everyone makes mistakes sometimes — it doesn’t mean you always fail. What matters is learning and growing."),

    # 🧠 Example 2
    ("human", 'Negative Thought: "Nobody likes me."'),
    ("ai", "It’s easy to feel that way when we’re down, but the truth is usually more nuanced. Some people do care about you, even if it's not obvious right now."),

    # 🧪 User Input (to be filled later)
    ("human", 'Negative Thought: "{negative_thought}"'),
])

# --- Chat History Management ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Display Past Chat ---
for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    st.chat_message(role).write(msg.content)

# --- Handle New User Input ---
user_input = st.chat_input("How can I help you with your wellness today?")

if user_input:
    st.chat_message("user").write(user_input)
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    # Format the prompt with user input
    prompt = cbt_prompt.format_messages(negative_thought=user_input)

    try:
        with st.spinner("Thinking Naturally ...."):
            response = model.invoke(prompt)
            st.chat_message("assistant").write(response.content)
            st.session_state.chat_history.append(response)
    except Exception as e:
        st.error(f"An error occurred: {e}")
