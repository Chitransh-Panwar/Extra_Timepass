from langgraph.graph import StateGraph,START,END
from typing import TypedDict,Annotated
from langchain_core.messages import BaseMessage,HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
import os
from dotenv import load_dotenv

class ChatState(TypedDict) :
    messages:Annotated[list[BaseMessage],add_messages]

load_dotenv()

llm=ChatOpenAI(
    model="openrouter/owl-alpha",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0.3,
    max_tokens=500
)
def chat_node(state:ChatState):
    messages=state['messages']
    response=llm.invoke(messages)
    return {'messages':[response]}

checkpointer= MemorySaver()

graph=StateGraph(ChatState)

graph.add_node('chat_node',chat_node)

graph.add_edge(START,'chat_node')
graph.add_edge('chat_node',END)

chatbot=graph.compile(checkpointer=checkpointer)

thread_id='1'
while True:
    user_message=input('User Type here : ')
    if user_message.strip().lower() in ['exit','quit','bye']:
        break
    config={'configurable':{'thread_id':thread_id}}
    response=chatbot.invoke({'messages':[HumanMessage(content=user_message)]},config=config)
    print('AI:',response['messages'][-1].content)