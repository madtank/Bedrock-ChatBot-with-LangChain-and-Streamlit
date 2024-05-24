import os
import streamlit as st
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import BedrockEmbeddings
from langchain_text_splitters import CharacterTextSplitter

def index_directory(directory_path, glob_pattern="**/[!.]*", uploaded_files=None):
    # Initialize Bedrock embeddings
    embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0")
    
    # Check if uploaded_files is not None and not empty
    if uploaded_files:
        documents = [file.getvalue().decode("utf-8") for file in uploaded_files]
    else:
        # Load documents from directory
        loader = DirectoryLoader(directory_path, glob=glob_pattern, show_progress=True, use_multithreading=True)
        documents = loader.load()

    # Split documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    # Check for existing FAISS index
    index_path = "faiss_index"
    if os.path.exists(index_path):
        vectorstore = FAISS.load_local(index_path, embeddings)
        vectorstore.add_documents(docs)
    else:
        # Create FAISS vectorstore
        vectorstore = FAISS.from_documents(docs, embeddings)

    # Save FAISS index locally
    vectorstore.save_local(index_path)

    return vectorstore

def main():
    st.title("Bedrock Indexer")
    uploaded_files = st.file_uploader("Choose files", accept_multiple_files=True)
    if st.button("Index Files"):
        if uploaded_files:
            vectorstore = index_directory(directory_path=None, uploaded_files=uploaded_files)
            st.success(f"Indexed {len(uploaded_files)} files. Total documents in index: {vectorstore.index.ntotal}")
        else:
            st.error("Please upload at least one file.")

if __name__ == "__main__":
    main()
