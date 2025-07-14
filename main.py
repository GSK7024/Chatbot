import os
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader, TextLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.vectorstores import chroma
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

required_env_vars = {
    
    "PINECONE_API_KEY": " PINECONE_API_KEY is not set."
}
   
for var, msg in required_env_vars.items():
    if var not in os.environ:
        raise ValueError(f"Error: {msg} Please set it as an environment variable.")

loaders = [
    (PyMuPDFLoader,"**/*.pdf"),
    (TextLoader,"**/*.txt")
]

documents = [] 
for loader_cls,global_pattern in loaders:
    loader = DirectoryLoader('C:/Users/gkgk7/OneDrive/Desktop/RAG_project/python-ai-project/documents',
    glob= global_pattern,
    loader_cls=loader_cls
    )
    
    loaded_docs = loader.load()

    if not loaded_docs:
        raise ValueError("File not found")
    
    documents.extend(loaded_docs)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunk = text_splitter.split_documents(documents)

embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

## Pinecone
# if "PINECONE_API_KEY" not in os.environ:
#     raise ValueError("PINECONE_API_KEY not found")
# index_name = "my_rag_project"
# vectorstore = PineconeVectorStore.from_documents(
#     chunk = documents,
#     embedding= embendding_model,
#     index_name = index_name 
# )
# retriever = vectorstore.as_retriever()

## Chroma_DB
vectorstore = Chroma.from_documents(
    documents=chunk ,
    embedding=embedding_model,
    persist_directory="./chroma_db"
)

retriever = vectorstore.as_retriever(search_kwargs={"k":1})

def get_rag_chain(provider="ollama"):
    llm = None
    if provider == "ollama":
        llm = ChatOllama(model="llama3:8b")
    elif provider == "openai":
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    elif provider == "claude":
        llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0)
    elif provider == "gemini":
        llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    template = """your helpful assistant, answer the questions based on the following context
{context}
question:{question}

ANSWER:
"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    chain = (
        {"context": lambda x: retriever.invoke(x["question"]), "question": lambda x: x["question"]}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

while True:
    provider_choice = input("Choose your AI provider (ollama, openai, claude, gemini): ").lower()
    if provider_choice in ["ollama", "openai", "claude", "gemini"]:
        break
    else:
        print("Invalid provider. Please choose again.")
print(f"Using {provider_choice.capitalize()} for this session.")
rag_chain = get_rag_chain(provider_choice)

while True:
    question = input("\\nYour Question (or type 'exit' to quit): ")
    if question.lower() == 'exit':
        print("Exiting.....")
        break
    
    response = rag_chain.invoke({"question": question})
    print(f"\\nAI :({provider_choice.capitalize()}): {response}")
