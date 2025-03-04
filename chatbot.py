from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
load_dotenv()

model = ChatGroq()
chat_history=[
    SystemMessage("you are a helpful AI assistant")
]

while True:
    user_input = input('You :')
    chat_history.append(HumanMessage(user_input))
    if user_input == 'exit':
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(result.content))
    print("AI:", result.content)
 
print(chat_history)