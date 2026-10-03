from datetime import datetime
from typing import Annotated, TypedDict
import sqlite3

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

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

tools = [calculator, get_current_datetime]


# ************************* Step 2: LLM ko tools batao *************************
# OpenAI key revoked hai, abhi Groq use kar rahe hain (GROQ_API_KEY .env se aata hai)
llm = ChatGroq(model='openai/gpt-oss-120b')
# bind_tools: LLM ko tools ki list (naam + docstring + args) bhejta hai.
# LLM khud tool nahi chalata, sirf bolta hai "ye tool in args ke saath chalao".
llm_with_tools = llm.bind_tools(tools)


# ************************* Step 3: graph *************************

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    """LLM ko poori history bhejo; jawab ya to text hoga ya tool call."""
    response = llm_with_tools.invoke(state['messages'])
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
