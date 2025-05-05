from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import os
from dotenv import load_dotenv

# Carrega variáveis do ambiente
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

dir_path = os.getenv("diretorio_arquivo")

# Carrega PDF
loader = PyPDFLoader(f"{dir_path}ácido zoledrônico_bula_1745598187754.pdf")
documents = loader.load()

# Divide texto em chunks menores
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
docs_split = text_splitter.split_documents(documents)

# Cria embeddings e salva num vetorstore (Chroma)
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

vectordb = Chroma.from_documents(
    documents=docs_split,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

vectordb.persist()

# Cria retriever
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# Define o modelo LLM correto
llm = ChatOpenAI(model_name="gpt-4", temperature=0)

# Pipeline RAG
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# Função simples para conversar com o RAG
def perguntar(pergunta):
    print(f"Pergunta: {pergunta}")
    resposta = rag_chain(pergunta)
    print("Resposta:", resposta['result'])
    
    # Fontes usadas
    print("\nFontes:")
    for doc in resposta["source_documents"]:
        print(f" - {doc.metadata['source']} (pág.: {doc.metadata.get('page', 'N/A')})")

# Exemplos de perguntas
perguntar("Do que se trata esse documento?")
perguntar("qual o tipo de uso dessa medicação?")
perguntar("Existem indicações de uso desta medicação?")
perguntar("Para quais tipos de câncer essa medicação serve?")
perguntar("Quais são as informações sobre os efeitos colaterais?")
