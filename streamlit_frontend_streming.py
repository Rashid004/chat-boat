import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from chatbot_backend import chatbot


# ************************* utility functions *************************

def generate_thread_id():
    """Har chat ke liye ek naya unique thread id (string) banata hai."""
    return str(uuid.uuid4())


def add_thread(thread_id):
    """Thread id ko sidebar ki list mein jodta hai (agar pehle se nahi hai)."""
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)


def reset_chat():
    """Nayi chat shuru karta hai: naya thread id, list mein add, screen khaali."""
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(thread_id)
    st.session_state['message_history'] = []


def load_conversation(thread_id):
    """Backend (checkpointer) se kisi thread ke saare messages nikalta hai."""
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    # naye thread mein 'messages' key nahi hoti, tab khaali list do
    return state.values.get('messages', [])


def to_ui_messages(messages):
    """LangChain messages ko UI format {'role', 'content'} mein badalta hai."""
    ui_messages = []
    for msg in messages:
        role = 'user' if isinstance(msg, HumanMessage) else 'assistant'
        ui_messages.append({'role': role, 'content': msg.content})
    return ui_messages


def ai_only_stream(user_input, config):
    """Chatbot ko stream mode mein chalata hai aur sirf AI ke tokens yield karta hai."""
    for message_chunk, metadata in chatbot.stream(
        {'messages': [HumanMessage(content=user_input)]},
        config=config,
        stream_mode='messages',
    ):
        if isinstance(message_chunk, AIMessage):
            yield message_chunk.content


# ************************* session setup *************************

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = []

add_thread(st.session_state['thread_id'])


# ************************* sidebar *************************

st.sidebar.title('Ladle Chatbot')

if st.sidebar.button('New Chat'):
    reset_chat()

st.sidebar.header('My Conversations')

# naye threads upar dikhane ke liye list ulti ([::-1])
for thread_id in st.session_state['chat_threads'][::-1]:
    if st.sidebar.button(thread_id, key=f'thread-{thread_id}'):
        st.session_state['thread_id'] = thread_id
        st.session_state['message_history'] = to_ui_messages(load_conversation(thread_id))


# ************************* main chat *************************

CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}

# purani chat history dikhana
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

user_input = st.chat_input('Type here')

if user_input:
    # user ka message save + display
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.markdown(user_input)

    # AI ka reply stream karke display
    with st.chat_message('assistant'):
        ai_message = st.write_stream(ai_only_stream(user_input, CONFIG))

    # AI ka reply history mein save (dobara display NAHI karna)
    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})
