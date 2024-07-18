import os
import chainlit as cl
from dotenv import load_dotenv
from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEndpoint
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyMuPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.runnable.config import RunnableConfig
from langchain_community.vectorstores import Qdrant
import tiktoken


# GLOBAL SCOPE - ENTIRE APPLICATION HAS ACCESS TO VALUES SET IN THIS SCOPE #
# ---- ENV VARIABLES ---- # 
"""
This function will load our environment file (.env) if it is present.

NOTE: Make sure that .env is in your .gitignore file - it is by default, but please ensure it remains there.
"""
load_dotenv()

"""
We will load our environment variables here.
"""

HF_TOKEN = os.environ["HF_TOKEN"]
HF_LLM_ENDPOINT = os.environ["HF_LLM_ENDPOINT"]
OPENAPI_KEY = os.environ["OPENAI_API_KEY"]

# ---- GLOBAL DECLARATIONS ---- #
#openai_client = ChatOpenAI(model="gpt-3.5-turbo")
# -- RETRIEVAL -- #
"""
1. Load Documents from Text File
2. Split Documents into Chunks
3. Load HuggingFace Embeddings (remember to use the URL we set above)
4. Index Files if they do not exist, otherwise load the vectorstore
"""
#Load the Pdf Documents from airbnb-10k    
documents = PyMuPDFLoader("data/airbnb-10k.pdf").load()

### 2. CREATE TEXT SPLITTER AND SPLIT DOCUMENTS
def tiktoken_len(text):
    tokens = tiktoken.encoding_for_model("gpt-4o").encode(
        text,
    )
    return len(tokens)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 0,
    length_function = tiktoken_len,
)

chunks = text_splitter.split_documents(documents)

#for chunk in chunks:
#  print(chunk)
#  print("----")

### 3. LOAD open ai EMBEDDINGS
HF_EMBED_ENDPOINT = os.environ["HF_EMBED_ENDPOINT"]

embeddings = HuggingFaceEndpointEmbeddings(
    model=HF_EMBED_ENDPOINT,
    task="feature-extraction",
    huggingfacehub_api_token=HF_TOKEN,
)


#Initialize the Vector Store
if os.path.exists(".vectorstore"):
    vectorstore = FAISS.load_local(
        "./vectorstore", 
        embeddings, 
        allow_dangerous_deserialization=True # this is necessary to load the vectorstore from disk as it's stored as a `.pkl` file.
    )
    retriever = vectorstore.as_retriever()
    print("Loaded Vectorstore")
else:
    print("Indexing Files")
    os.makedirs("./vectorstore", exist_ok=True)
    for i in range(0, len(chunks), 32):
        if i == 0:
            vectorstore = FAISS.from_documents(chunks[i:i+32], embeddings)
            continue
        vectorstore.add_documents(chunks[i:i+32])
    vectorstore.save_local("./vectorstore")
    
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    
# -- AUGMENTED -- #
"""
1. Define a String Template
2. Create a Prompt Template from the String Template
"""
### 1. DEFINE STRING TEMPLATE
RAG_PROMPT_TEMPLATE = """\
<|start_header_id|>system<|end_header_id|>
You are a helpful assistant. You answer user questions based on provided context. If you can't answer the question with the provided context, say you don't know.<|eot_id|>

<|start_header_id|>user<|end_header_id|>
User Query:
{query}

Context:
{context}<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
"""

### 2. CREATE PROMPT TEMPLATE
rag_prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

# -- GENERATION -- #
"""
1. Create a HuggingFaceEndpoint for the LLM
"""
hf_llm = HuggingFaceEndpoint(
    endpoint_url=HF_LLM_ENDPOINT,
    max_new_tokens=512,
    top_k=2,
    top_p=0.95,
    typical_p=0.95,
    temperature=0.01,
    repetition_penalty=1.03,
    huggingfacehub_api_token=HF_TOKEN,
)

print(hf_llm.invoke("What is Deep Learning?"))


@cl.on_chat_start
async def start_chat():
    """
    This function will be called at the start of every user session. 

    We will build our LCEL RAG chain here, and store it in the user session. 

    The user session is a dictionary that is unique to each user session, and is stored in the memory of the server.
    """

    ### BUILD LCEL RAG CHAIN THAT ONLY RETURNS TEXT
    lcel_rag_chain = (
        # INVOKE CHAIN WITH: {"question" : "<<SOME USER QUESTION>>"}
        # "question" : populated by getting the value of the "question" key
        # "context"  : populated by getting the value of the "question" key and chaining it into the base_retriever
        {"context": itemgetter("query") | retriever, "query": itemgetter("query")}
        # "context"  : is assigned to a RunnablePassthrough object (will not be called or considered in the next step)
        #              by getting the value of the "context" key from the previous step
        | RunnablePassthrough.assign(context=itemgetter("context"))
        # "response" : the "context" and "question" values are used to format our prompt object and then piped
        #              into the LLM and stored in a key called "response"
        # "context"  : populated by getting the value of the "context" key from the previous step
        | {"response": rag_prompt | hf_llm, "context": itemgetter("context")}
    )

    cl.user_session.set("lcel_rag_chain", lcel_rag_chain)

@cl.on_message  
async def main(message: cl.Message):
    """
    This function will be called every time a message is recieved from a session.

    We will use the LCEL RAG chain to generate a response to the user query.

    The LCEL RAG chain is stored in the user session, and is unique to each user session - this is why we can access it here.
    """
    lcel_rag_chain = cl.user_session.get("lcel_rag_chain")

    response = lcel_rag_chain.invoke({"query" : message.content})

    await cl.Message(content=response).send()