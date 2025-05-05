# !pip install langchain

import getpass
import os
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage


# Carrega variáveis do ambiente
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
#os.environ["LANGSMITH_TRACING"] = "true"


model = init_chat_model("gpt-4o-mini", model_provider="openai")
messages = [
    SystemMessage("Informe quais os nomes comerciais da medicação abaixo"),
    HumanMessage("Diazepam"),
]

result = model.invoke(messages)
print(result)