import html
import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from chatbot_backend import chatbot, retrieve_all_threads


# ************************* page config + styles *************************

st.set_page_config(page_title='JPT', page_icon=':material/forum:', layout='centered')

# empty screen par dikhne wale ready-made prompts
SUGGESTIONS = [
    'Explain LangGraph in simple words',
    'Write a Python function to reverse a string',
    'Give me 3 tips to learn faster',
    'Summarize what a checkpointer does',
]

# Lucide "sparkles" icon (emoji ki jagah SVG)
SPARKLES_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" '
    'fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" '
    'stroke-linejoin="round" aria-hidden="true">'
    '<path d="M9.937 15.5A2 2 0 0 0 8.5 14.063l-6.135-1.582a.5.5 0 0 1 0-.962L8.5 9.936'
    'A2 2 0 0 0 9.937 8.5l1.582-6.135a.5.5 0 0 1 .963 0L14.063 8.5A2 2 0 0 0 15.5 9.937'
    'l6.135 1.581a.5.5 0 0 1 0 .964L15.5 14.063a2 2 0 0 0-1.437 1.437l-1.582 6.135'
    'a.5.5 0 0 1-.963 0z"/><path d="M20 3v4"/><path d="M22 5h-4"/>'
    '<path d="M4 17v2"/><path d="M5 18H3"/></svg>'
)

TYPING_HTML = (
    '<div class="typing" role="status" aria-label="Assistant is typing">'
    '<span></span><span></span><span></span></div>'
)

CUSTOM_CSS = """
<style>
/* ---------- color tokens (config.toml ke saath match) ---------- */
:root {
    --bg: #0E0E14;
    --surface: #16161F;
    --surface-hover: #1E1E2A;
    --border: #262633;
    --text: #ECECF4;
    --text-muted: #9A9AB0;
    --primary: #7C3AED;
    --primary-soft: #A78BFA;
    --primary-bg: #221B3A;
    --primary-border: #3B2F66;
    --gradient: linear-gradient(135deg, #7C3AED, #0891B2);
}

/* ---------- general ---------- */
header[data-testid="stHeader"] { background: transparent; }
[data-testid="stMainBlockContainer"] { padding-top: 2.5rem; padding-bottom: 1.5rem; }
button:focus-visible { outline: 2px solid var(--primary-soft); outline-offset: 2px; }

/* ---------- sidebar ---------- */
.brand { display: flex; align-items: center; gap: .65rem; margin: .25rem 0 1rem; }
.brand-logo {
    width: 34px; height: 34px; border-radius: 10px; display: grid; place-items: center;
    color: #fff; background: var(--gradient);
}
.brand-name { font-size: 1.1rem; font-weight: 700; color: var(--text); }
.brand-sub { font-size: .75rem; color: var(--text-muted); }
.section-label {
    font-size: .72rem; font-weight: 600; letter-spacing: .06em; text-transform: uppercase;
    color: var(--text-muted); margin: 1.25rem 0 .25rem .35rem;
}

/* conversation list items */
[class*="st-key-thread-"] button {
    justify-content: flex-start; min-height: 40px; padding: .45rem .7rem;
    border-radius: 10px; color: var(--text-muted); transition: background-color .15s ease, color .15s ease;
}
[class*="st-key-thread-"] button > div { justify-content: flex-start; min-width: 0; width: 100%; }
[class*="st-key-thread-"] button p {
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; text-align: left;
}
[class*="st-key-thread-"] [data-testid="stBaseButton-tertiary"]:hover {
    background: var(--surface-hover); color: var(--text);
}
[class*="st-key-thread-"] [data-testid="stBaseButton-secondary"] {
    background: var(--primary-bg); border-color: var(--primary-border); color: #DDD6FE; font-weight: 600;
}

/* ---------- empty state ---------- */
.hero { text-align: center; margin: 10vh 0 2rem; }
.hero-icon {
    width: 56px; height: 56px; margin: 0 auto 1rem; border-radius: 16px;
    display: grid; place-items: center; color: #fff; background: var(--gradient);
    box-shadow: 0 10px 40px rgba(124, 58, 237, .35);
}
.hero-title { font-size: 1.85rem; font-weight: 700; color: var(--text); line-height: 1.25; }
.hero-sub { color: var(--text-muted); margin-top: .4rem; }

[class*="st-key-suggest-"] button {
    justify-content: flex-start; text-align: left; min-height: 56px; padding: .75rem 1rem;
    background: var(--surface); border: 1px solid var(--border); color: var(--text);
    transition: border-color .15s ease, background-color .15s ease;
}
[class*="st-key-suggest-"] button > div { justify-content: flex-start; }
[class*="st-key-suggest-"] button:hover {
    background: var(--surface-hover); border-color: var(--primary); color: var(--text);
}

/* ---------- chat messages ---------- */
[data-testid="stChatMessage"] { background: transparent; padding: .35rem 0; gap: .75rem; }
[data-testid="stChatMessageContent"] { line-height: 1.65; }

/* user message: right side bubble, avatar hidden */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    width: fit-content; max-width: 85%; margin-left: auto;
}
[data-testid="stChatMessageAvatarUser"] { display: none; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
    background: var(--primary-bg); border: 1px solid var(--primary-border);
    border-radius: 18px 18px 4px 18px; padding: .55rem 1rem;
}

/* assistant avatar: brand gradient */
[data-testid="stChatMessageAvatarAssistant"] { background: var(--gradient); color: #fff; }

/* ---------- chat input ---------- */
[data-testid="stChatInput"] {
    border-radius: 16px; background: var(--surface); border-color: var(--border);
    transition: border-color .15s ease, box-shadow .15s ease;
}
[data-testid="stChatInput"]:focus-within {
    border-color: var(--primary); box-shadow: 0 0 0 3px rgba(124, 58, 237, .25);
}

/* ---------- typing indicator ---------- */
.typing { display: inline-flex; align-items: center; gap: 5px; padding: .7rem 0; }
.typing span {
    width: 7px; height: 7px; border-radius: 50%; background: var(--primary-soft);
    animation: typing 1.2s infinite ease-in-out;
}
.typing span:nth-child(2) { animation-delay: .15s; }
.typing span:nth-child(3) { animation-delay: .3s; }
@keyframes typing {
    0%, 80%, 100% { opacity: .35; transform: translateY(0); }
    40% { opacity: 1; transform: translateY(-3px); }
}

/* ---------- tool status pill (web search etc.) ---------- */
.status-pill {
    display: inline-flex; align-items: center; gap: .6rem; margin: .45rem 0;
    padding: .4rem .9rem .4rem .75rem; border-radius: 999px; font-size: .86rem;
    background: var(--primary-bg); border: 1px solid var(--primary-border);
}
.status-pill .spin {
    width: 14px; height: 14px; border-radius: 50%; flex: none;
    border: 2px solid var(--primary-border); border-top-color: var(--primary-soft);
    animation: spin .8s linear infinite;
}
.status-pill .status-text {
    background: linear-gradient(90deg, var(--text-muted) 20%, var(--text) 50%, var(--text-muted) 80%);
    background-size: 200% 100%; -webkit-background-clip: text; background-clip: text;
    color: transparent; animation: shimmer 1.8s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }
@keyframes shimmer { from { background-position: 200% 0; } to { background-position: -200% 0; } }

@media (prefers-reduced-motion: reduce) {
    .typing span { animation: none; opacity: .6; }
    .status-pill .spin { animation: none; }
    .status-pill .status-text { animation: none; background: none; color: var(--text); }
    * { transition: none !important; }
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


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
    # current chat pehle se khaali hai to ek aur khaali chat mat banao
    if not st.session_state['message_history']:
        return
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(thread_id)
    st.session_state['message_history'] = []


def switch_thread(thread_id):
    """Sidebar se purani chat kholta hai aur uske messages screen par laata hai."""
    st.session_state['thread_id'] = thread_id
    st.session_state['message_history'] = to_ui_messages(load_conversation(thread_id))


def use_suggestion(text):
    """Suggestion card click hone par us text ko agle message ki tarah bhejta hai."""
    st.session_state['pending_prompt'] = text


def load_conversation(thread_id):
    """Backend (checkpointer) se kisi thread ke saare messages nikalta hai."""
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    # naye thread mein 'messages' key nahi hoti, tab khaali list do
    return state.values.get('messages', [])


def to_ui_messages(messages):
    """LangChain messages ko UI format {'role', 'content'} mein badalta hai."""
    ui_messages = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            ui_messages.append({'role': 'user', 'content': msg.content})
        # sirf asli AI jawab dikhao; tool call wale (khaali content) aur ToolMessage chhupao
        elif isinstance(msg, AIMessage) and msg.content:
            ui_messages.append({'role': 'assistant', 'content': msg.content})
    return ui_messages


def get_thread_title(thread_id, max_len=32):
    """Thread ka pehla user message hi uska title hai (lamba ho to kaat do)."""
    for msg in load_conversation(thread_id):
        if isinstance(msg, HumanMessage):
            title = msg.content.strip()
            return title if len(title) <= max_len else title[:max_len].rstrip() + '…'
    return 'New chat'


TOOL_LABELS = {
    'web_search': 'Searching the web',
    'calculator': 'Calculating',
    'get_current_datetime': 'Checking the time',
}


def status_html(label):
    return (
        '<div class="status-pill" role="status" aria-live="polite" aria-busy="true">'
        f'<span class="spin"></span><span class="status-text">{html.escape(label)}…</span></div>'
    )


def ai_only_stream(user_input, config, on_status):
    """Chatbot ko stream mode mein chalata hai: AI ke tokens yield karta hai,
    tool chalne par on_status(label) bulata hai."""
    for message_chunk, metadata in chatbot.stream(
        {'messages': [HumanMessage(content=user_input)]},
        config=config,
        stream_mode='messages',
    ):
        if isinstance(message_chunk, ToolMessage):
            on_status('Reading sources' if message_chunk.name == 'web_search' else 'Writing the answer')
        elif isinstance(message_chunk, AIMessage):
            for call in message_chunk.tool_call_chunks or []:
                if call.get('name'):  # tool call ka pehla chunk naam leke aata hai
                    on_status(TOOL_LABELS.get(call['name'], 'Working'))
            if message_chunk.content:
                yield message_chunk.content


def hide_on_first_token(stream, placeholder, state):
    """Pehla asli token aate hi status/typing indicator hata deta hai."""
    for token in stream:
        if token and not state['started']:
            state['started'] = True
            placeholder.empty()
        yield token


# ************************* session setup *************************

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrieve_all_threads()

add_thread(st.session_state['thread_id'])


# ************************* sidebar *************************

with st.sidebar:
    st.markdown(
        f'<div class="brand"><div class="brand-logo">{SPARKLES_SVG.format(size=18)}</div>'
        '<div><div class="brand-name">JPT</div>'
        '<div class="brand-sub">AI assistant</div></div></div>',
        unsafe_allow_html=True,
    )
    st.button('New chat', icon=':material/edit_square:', type='primary', width='stretch', on_click=reset_chat)
    st.markdown('<div class="section-label">Recent</div>', unsafe_allow_html=True)
    # list script ke end mein bharenge, taaki naye message ka title turant dikhe
    thread_list = st.container()


# ************************* main chat *************************

# configurable: checkpointer ke liye; metadata + run_name: LangSmith trace ke liye
CONFIG = {
    'configurable': {'thread_id': st.session_state['thread_id']},
    'metadata': {'thread_id': st.session_state['thread_id']},
    'run_name': 'chat_turn',
}

user_input = st.chat_input('Message JPT…')
# chat box ya suggestion card, jo bhi aaya ho
prompt = user_input or st.session_state.pop('pending_prompt', None)

# khaali chat: welcome screen + suggestions
if not st.session_state['message_history'] and not prompt:
    st.markdown(
        f'<div class="hero"><div class="hero-icon">{SPARKLES_SVG.format(size=26)}</div>'
        '<div class="hero-title" role="heading" aria-level="1">How can I help you today?</div>'
        '<div class="hero-sub">Ask anything, or start with one of these.</div></div>',
        unsafe_allow_html=True,
    )
    cols = st.columns(2)
    for i, text in enumerate(SUGGESTIONS):
        cols[i % 2].button(text, key=f'suggest-{i}', width='stretch', on_click=use_suggestion, args=(text,))

# purani chat history dikhana
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

if prompt:
    # user ka message save + display
    st.session_state['message_history'].append({'role': 'user', 'content': prompt})
    with st.chat_message('user'):
        st.markdown(prompt)

    # AI ka reply stream karke display (pehle typing dots, phir tokens)
    with st.chat_message('assistant'):
        typing = st.empty()
        typing.markdown(TYPING_HTML, unsafe_allow_html=True)
        state = {'started': False}

        def on_status(label):
            if not state['started']:
                typing.markdown(status_html(label), unsafe_allow_html=True)

        try:
            stream = hide_on_first_token(ai_only_stream(prompt, CONFIG, on_status), typing, state)
            ai_message = st.write_stream(stream)
        except Exception as error:
            typing.empty()
            st.error("Couldn't get a reply right now. Please try again.", icon=':material/error:')
            st.caption(f'{type(error).__name__}: {error}')
            ai_message = None

    # AI ka reply history mein save (dobara display NAHI karna)
    if ai_message:
        st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})


# ************************* sidebar: conversation list *************************

with thread_list:
    # naye threads upar dikhane ke liye list ulti ([::-1])
    for thread_id in st.session_state['chat_threads'][::-1]:
        is_active = thread_id == st.session_state['thread_id']
        st.button(
            get_thread_title(thread_id),
            key=f'thread-{thread_id}',
            type='secondary' if is_active else 'tertiary',
            icon=':material/chat_bubble:',
            width='stretch',
            on_click=switch_thread,
            args=(thread_id,),
        )
