import os
from openai import OpenAI
from langsmith.wrappers import wrap_openai
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"]
#LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
#LANGSMITH_PROJECT="pr-charming-recreation-77"
from langsmith import traceable

# trace apenas na chamada da openai
openai_client = wrap_openai(OpenAI())


# para tracear um método ou aplicação toda:



# This is the retriever we will use in RAG
# This is mocked out, but it could be anything we want
def retriever(query: str):
    results = ["Harrison worked at Kensho for 10 years. To be exactly: from 2012 to 2022"]
    return results


@traceable(run_type="retriever")
def traceable_retriever(query: str):
    results = ["Harrison worked at Kensho"]
    return results


# This is the end-to-end RAG chain.
# It does a retrieval step then calls OpenAI
def rag(question):
    docs = retriever(question)
    system_message = """Answer the users question using only the provided information below:
    
    {docs}""".format(docs="\n".join(docs))
    
    return openai_client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": question},
        ],
        model="gpt-4o-mini",
    )

@traceable
def traceable_rag(question):
    docs = traceable_retriever(question)
    system_message = """Answer the users question using only the provided information below:
    
    {docs}""".format(docs="\n".join(docs))
    
    return openai_client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": question},
        ],
        model="gpt-4o-mini",
    )



#rag("where did harrison work")
#rag("how time harrison worked in kensho?")

traceable_rag("where did harrison work")
traceable_rag("how time harrison worked in kensho?")

