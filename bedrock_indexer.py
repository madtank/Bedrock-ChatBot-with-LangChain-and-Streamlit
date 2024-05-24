from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import BedrockEmbeddings
from langchain_text_splitters import CharacterTextSplitter
import os

def index_directory(directory_path, vectorstore_path="vectorstore", glob_pattern="**/[!.]*"):
    # Initialize Bedrock embeddings
    embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0")
    # Load documents from directory
    loader = DirectoryLoader(directory_path, glob=glob_pattern, show_progress=True, use_multithreading=True)
    documents = loader.load()

    # Split documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    # Check if a vectorstore already exists and load it; otherwise, create a new one
    if os.path.exists(vectorstore_path):
        vectorstore = FAISS.load_local(vectorstore_path, embeddings)
    else:
        vectorstore = FAISS(embeddings)

    # Add documents to the vectorstore
    vectorstore.add_documents(docs)

    # Save FAISS index locally
    vectorstore.save_local(vectorstore_path)

    return vectorstore

# Example usage
directory_path = "documents/"
vectorstore_path = "vectorstore"
vectorstore = index_directory(directory_path, vectorstore_path)
print(f"Total documents indexed: {vectorstore.index.ntotal}")
