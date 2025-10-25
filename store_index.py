from dotenv import load_dotenv
import os
from src.helper import load_pdf_files,filter_to_minimal_docs,text_slipt,get_embeddings  
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()
PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
GROQ_API_KEY=os.getenv("GROQ_API_KEY")

os.environ["PINECONE_API_KEY"]=PINECONE_API_KEY  
os.environ["GROQ_API_KEY"]=GROQ_API_KEY

extracted_docs=load_pdf_files(data="data/")
minimal_docs=filter_to_minimal_docs(extracted_docs)
text_chunks=text_slipt(minimal_docs)

embeddings=get_embeddings()

Pinecone_api_key=PINECONE_API_KEY
pc  = Pinecone(api_key=Pinecone_api_key)

index_name="medical-chatbot"
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws",region="us-east-1")     
    )
index=pc.Index(index_name)


vector_store = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=index_name,
  
)

