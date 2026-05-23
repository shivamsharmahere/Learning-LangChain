from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(
    model= "llama-3.1-8b-instant",
    temperature=0.5
)

chat_history = []
#load chat_history
with open('chat_history.txt') as f:
    chat_history.extend(f.readlines())

template = ChatPromptTemplate([
    ('system', "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    ('human', '{input}')
])

chat_template = template.invoke(chat_history=chat_history, input="{input}")  

while True:
    user_input = input("You: ")
    chat_history.append(f"Human: {user_input}\n")
    if user_input.lower() in ["exit", "quit"]:
        print("AI: Goodbye!")
        print("Chat history:", chat_history)
        break
    response = model.invoke( )
    chat_history.append(f"AI: {response.content}\n")
    print("AI:", response.content)
