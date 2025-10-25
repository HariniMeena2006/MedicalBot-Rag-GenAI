from langchain.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain.schema import Document
 #Extract Data from PDF files
def load_pdf_files(data):
    loader=DirectoryLoader(data,
                           glob="*.pdf",
                           loader_cls=PyPDFLoader)
    documents=loader.load()
    return documents
# filter documents to minimal metadata

def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Given a list of Document objects, return a new list of Document objects
    containing only 'source' in metadata and the original page_content.
    """
    minimal_docs: List[Document] = []

    for doc in docs:
        # Get the 'source' field from metadata
        src = doc.metadata.get("source")

        # Create a new minimal Document with only 'source' metadata
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )

    return minimal_docs
#spilt to chunks

def text_slipt(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
        length_function=len,
    )
    text_chunks = text_splitter.split_documents(minimal_docs)
    return text_chunks
#download the embedding model
from langchain.embeddings import HuggingFaceEmbeddings
def get_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    return embeddings




