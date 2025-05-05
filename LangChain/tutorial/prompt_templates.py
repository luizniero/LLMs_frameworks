import getpass
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
# Cria um prompt.
system_template = "Translate the following from English into {language}"

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)


prompt = prompt_template.invoke({"language": "Italian", "text": "hi!"})
#print(prompt)
print(prompt.to_messages())


model = init_chat_model("gpt-4o-mini", model_provider="openai")
result = model.invoke(prompt)
print(result)
print(f"Resposta: {result.content}")