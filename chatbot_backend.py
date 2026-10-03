from datetime import datetime
from typing import Annotated, Literal, TypedDict
import sqlite3

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_tavily import TavilySearch

load_dotenv()


# ************************* Step 1: tools *************************
# @tool function ko LLM ke liye "tool" bana deta hai.
# Docstring + type hints hi LLM ko batate hain ki tool kab aur kaise use karna hai.

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """Do basic arithmetic on two numbers. operation must be one of: add, sub, mul, div."""
    if operation == 'add':
        result = first_num + second_num
    elif operation == 'sub':
        result = first_num - second_num
    elif operation == 'mul':
        result = first_num * second_num
    elif operation == 'div':
        if second_num == 0:
            return {'error': 'Division by zero is not allowed'}
        result = first_num / second_num
    else:
        return {'error': f'Unsupported operation: {operation}'}
    return {'first_num': first_num, 'second_num': second_num, 'operation': operation, 'result': result}


@tool
def get_current_datetime() -> str:
    """Get the current local date and time. Use this for any question about today, now, or the time."""
    return datetime.now().strftime('%A, %d %B %Y, %I:%M %p')

# TavilySearch ke ~10 optional args se gpt-oss confuse hota hai (galat tool call bhejta hai).
# Isliye chhota wrapper: LLM ko sirf 2 args dikhte hain, aur topic wo khud chunta hai.
_tavily = TavilySearch(max_results=3)


@tool
def web_search(query: str, topic: Literal['general', 'news'] = 'general') -> dict:
    """Search the web for current information. Use topic='news' for news and current events,
    topic='general' for everything else (facts, prices, sports, weather)."""
    return _tavily.invoke({'query': query, 'topic': topic})


tools = [calculator, get_current_datetime, web_search]


# ************************* Step 2: LLM ko tools batao *************************
# OpenAI key revoked hai, abhi Groq use kar rahe hain (GROQ_API_KEY .env se aata hai)
llm = ChatGroq(model='openai/gpt-oss-120b')
# bind_tools: LLM ko tools ki list (naam + docstring + args) bhejta hai.
# LLM khud tool nahi chalata, sirf bolta hai "ye tool in args ke saath chalao".
llm_with_tools = llm.bind_tools(tools)


# ************************* Step 3: graph *************************

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# system prompt har call par judta hai, par state/DB mein save nahi hota
SYSTEM_PROMPT = SystemMessage(content=(
    "You are JPT, a helpful assistant. "
    "Use web_search when the answer may depend on recent or changing information "
    "(news, current events, prices, sports, weather, recent facts) or when you are unsure. "
    "For 'today', 'latest', 'breaking' or 'current' questions use topic='news'. "
    "Answer directly, without searching, for general knowledge, coding, or casual chat. "
    "Never say you lack real-time access; search instead. Mention the source when you use search results."
))


def chat_node(state: ChatState):
    """LLM ko poori history bhejo; jawab ya to text hoga ya tool call."""
    response = llm_with_tools.invoke([SYSTEM_PROMPT] + state['messages'])
    return {'messages': [response]}


# ToolNode: LLM ne jo tool calls maange, unhe chala ke ToolMessage return karta hai
tool_node = ToolNode(tools)

connection = sqlite3.connect(database='chatbot.db', check_same_thread=False)

# Check Pointer
checkpointer = SqliteSaver(conn=connection)

graph = StateGraph(ChatState)

graph.add_node('chat_node', chat_node)
graph.add_node('tools', tool_node)

graph.add_edge(START, 'chat_node')
# tools_condition: last message mein tool call hai to 'tools' par jao, warna END
graph.add_conditional_edges('chat_node', tools_condition)
# tool ka result wapas LLM ko, taaki wo final jawab likh sake (yahi loop hai)
graph.add_edge('tools', 'chat_node')

chatbot = graph.compile(checkpointer=checkpointer)


def retrieve_all_threads():
    """DB mein saved saare thread ids, purane se naye order mein."""
    all_threads = []
    # list() naye checkpoints pehle deta hai; set() order kho deta, isliye list
    for checkpoint in checkpointer.list(None):
        thread_id = checkpoint.config['configurable']['thread_id']
        if thread_id not in all_threads:
            all_threads.append(thread_id)

    # frontend [::-1] karta hai, isliye yahan purane pehle rakho
    return all_threads[::-1]
