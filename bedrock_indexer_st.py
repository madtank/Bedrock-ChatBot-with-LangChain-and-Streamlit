# filename: index_files.py  
import streamlit as st  
import numpy as np  
import os  
from langchain_community.vectorstores import FAISS  
from langchain_community.embeddings import BedrockEmbeddings  
from langchain_text_splitters import RecursiveCharacterTextSplitter  
import PyPDF2  

# Create a global variable for the embeddings
EMBEDDINGS = BedrockEmbeddings(model_id="cohere.embed-english-v3")  

def index_file(uploaded_files=None):  
    """  
    Indexes the uploaded files using Bedrock embeddings and FAISS vector store.  
  
    Args:  
        uploaded_files (list): List of uploaded files.  
  
    Returns:  
        tuple: A tuple containing the documents and their combined embeddings.  
    """  
    if uploaded_files:  
        documents = []  
        for file in uploaded_files:  
            if file.type == "application/pdf":  
                pdf_reader = PyPDF2.PdfReader(file)  
                document = ""  
                for page in range(len(pdf_reader.pages)):  
                    document += pdf_reader.pages[page].extract_text()  
            else:  
                document = file.getvalue().decode("utf-8")  
            documents.append(document)  
  
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)  
        docs = text_splitter.create_documents(documents)  
  
        document_embeddings = EMBEDDINGS.embed_documents([doc.page_content for doc in docs])  
  
        combined_embeddings = np.array(document_embeddings)  
        if len(combined_embeddings.shape) == 1:  
            combined_embeddings = combined_embeddings.reshape(-1, 1)  
  
        return docs, combined_embeddings  
    else:  
        return None, None  
  
def rag_search(prompt: str, index_path) -> list:  
    allow_dangerous = True  
    db = FAISS.load_local(index_path, EMBEDDINGS, allow_dangerous_deserialization=allow_dangerous)  
    docs = db.similarity_search(prompt, k=5)  
    return docs

def search_index(index_path):  
    query = st.session_state["search_query"]  
    if query:  
        matching_docs = rag_search(query, index_path)  
        st.session_state["matching_docs"] = matching_docs  
    else:  
        st.error("Please enter a search query.")  

def main():  
    index_path = "faiss_index"  
    uploaded_files = st.file_uploader("Choose files", accept_multiple_files=True)  
  
    if st.button("Index Files"):  
        if uploaded_files:  
            docs, combined_embeddings = index_file(uploaded_files)  
            if docs is None or combined_embeddings is None:  
                return  
            if os.path.exists(index_path):  
                vectorstore = FAISS.load_local(index_path, EMBEDDINGS, allow_dangerous_deserialization=True)  
                vectorstore.add_documents(docs)  
                vectorstore.save_local(index_path)  
            else:  
                vectorstore = FAISS.from_documents(docs, EMBEDDINGS)  
                vectorstore.save_local(index_path)  
            st.success(f"{len(uploaded_files)} files indexed. Total documents in index: {vectorstore.index.ntotal}")  
        else:  
            st.error("Please upload files before indexing.")  
  
    if "search_query" not in st.session_state:  
        st.session_state["search_query"] = ""  
  
    st.text_input("Enter your search query", key="search_query", on_change=search_index, args=(index_path,))  
  
    # Create the placeholder at the point where you want the search results to appear
    placeholder = st.empty()
    # Move the display of search results and indexed documents to the bottom
    if "matching_docs" in st.session_state:  
        with placeholder.container():  
            formatted_docs = "\n\n---\n\n".join(str(doc) for doc in st.session_state['matching_docs'])
            st.markdown(f"## Search Results\n\n{formatted_docs}")

if __name__ == "__main__":  
    main()