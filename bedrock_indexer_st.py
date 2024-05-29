# filename: index_files.py  
import streamlit as st  
import numpy as np  
import os  
from langchain_community.vectorstores import FAISS  
from langchain_community.embeddings import BedrockEmbeddings  
from langchain_text_splitters import RecursiveCharacterTextSplitter  
import PyPDF2  
  
def extract_metadata(file):  
    """  
    Extracts metadata from the uploaded file.  
  
    Args:  
        file: The uploaded file.  
  
    Returns:  
        dict: A dictionary containing metadata.  
    """  
    metadata = {}  
    if file.type == "application/pdf":  
        pdf_reader = PyPDF2.PdfReader(file)
        metadata["title"] = pdf_reader.getDocumentInfo().title
        metadata["author"] = pdf_reader.getDocumentInfo().author
    else:  
        metadata["title"] = file.name
        metadata["author"] = "Unknown"  
    return metadata
  
def classify_document(content):  
    """  
    Placeholder function for AI classification of the document content.  
  
    Args:  
        content (str): The document content.  
  
    Returns:  
        str: The classification label.  
    """  
    # Placeholder for manual classification  
    # You can implement your own classification logic here  
    return "Unclassified"  
  
def index_file(uploaded_files=None):  
    """  
    Indexes the uploaded files using Bedrock embeddings and FAISS vector store.  
  
    Args:  
        uploaded_files (list): List of uploaded files.  
  
    Returns:  
        tuple: A tuple containing the documents and their combined embeddings.  
    """  
    embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0")  
      
    if uploaded_files:  
        documents = []  
        for file in uploaded_files:  
            metadata = extract_metadata(file)  
            if file.type == "application/pdf":  
                pdf_reader = PyPDF2.PdfReader(file)  
                document = ""  
                for page in range(len(pdf_reader.pages)):  
                    document += pdf_reader.pages[page].extract_text()  
                metadata["content"] = document  
            else:  
                metadata["content"] = file.getvalue().decode("utf-8")  
            metadata["classification"] = classify_document(metadata["content"])  
            documents.append(metadata)  
  
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)  
        docs = text_splitter.create_documents([doc["content"] for doc in documents])  
  
        document_embeddings = embeddings.embed_documents([doc.page_content for doc in docs])  
  
        combined_embeddings = np.array(document_embeddings)  
        if len(combined_embeddings.shape) == 1:  
            combined_embeddings = combined_embeddings.reshape(-1, 1)  
  
        for i, doc in enumerate(docs):  
            doc.metadata = documents[i]  
  
        return docs, combined_embeddings  
    else:  
        return None, None  
  
def rag_search(prompt: str, index_path) -> str:  
    embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0")  
    allow_dangerous = True  
    db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=allow_dangerous)  
    docs = db.similarity_search(prompt, k=5)  
    rag_content = "Here are the RAG search results: \n\n<search>\n\n" + "\n\n".join(doc.page_content for doc in docs) + "\n\n</search>\n\n"  
    return rag_content + prompt  
  
def search_index(index_path):  
    query = st.session_state["search_query"]  
    if query:  
        matching_docs = rag_search(query, index_path)  
        st.session_state["matching_docs"] = matching_docs  
    else:  
        st.error("Please enter a search query.")  
  
def view_index(index_path):  
    embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0")  
    allow_dangerous = True  
    db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=allow_dangerous)  
    docs = db.afrom_documents
    return docs  
  
def delete_document(index_path, doc_id):  
    embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0")  
    allow_dangerous = True  
    db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=allow_dangerous)  
    db.delete_document(doc_id)  
    db.save_local(index_path)  
  
def main():  
    index_path = "faiss_index"  
    placeholder = st.empty()  
  
    uploaded_files = st.file_uploader("Choose files", accept_multiple_files=True)  
    if st.button("Index Files"):  
        if uploaded_files:  
            docs, combined_embeddings = index_file(uploaded_files)  
            if docs is None or combined_embeddings is None:  
                return  
            if os.path.exists(index_path):  
                vectorstore = FAISS.load_local(index_path, BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0"), allow_dangerous_deserialization=True)  
                vectorstore.add_documents(docs)  
                vectorstore.save_local(index_path)  
            else:  
                vectorstore = FAISS.from_documents(docs, BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0"))  
                vectorstore.save_local(index_path)  
            st.success(f"{len(uploaded_files)} files indexed. Total documents in index: {vectorstore.index.ntotal}")  
        else:  
            st.error("Please upload at least one file.")  
  
    if "search_query" not in st.session_state:  
        st.session_state["search_query"] = ""  
  
    st.text_input("Enter your search query", key="search_query", on_change=search_index, args=(index_path,))  
  
    if "matching_docs" in st.session_state:  
        with placeholder.container():  
            st.write(st.session_state["matching_docs"])  
  
    if st.button("View Indexed Documents"):  
        docs = view_index(index_path)  
        for doc in docs:  
            st.write(f"ID: {doc.id}, Title: {doc.metadata['title']}, Author: {doc.metadata['author']}")  
  
    doc_id_to_delete = st.text_input("Enter Document ID to Delete")  
    if st.button("Delete Document"):  
        if doc_id_to_delete:  
            delete_document(index_path, doc_id_to_delete)  
            st.success(f"Document {doc_id_to_delete} deleted.")  
        else:  
            st.error("Please enter a document ID.")  
  
if __name__ == "__main__":  
    main()  