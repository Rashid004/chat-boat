from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import sqlite3

load_dotenv()

# OpenAI key revoked hai, abhi Groq use kar rahe hain (GROQ_API_KEY .env se aata hai)
llm = ChatGroq(model='openai/gpt-oss-120b')

class ChatState(TypedDict):
  messages: Annotated[list[BaseMessage], add_messages]
  
  
def chat_node(state: ChatState):
  messages = state["messages"]
  response = llm.invoke(messages)
  return {"messages": [response]}

connection = sqlite3.connect(database='chatbot.db', check_same_thread=False)

# Check Pointer
checkpointer = SqliteSaver(conn=connection)

graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

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
