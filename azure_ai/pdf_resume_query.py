import os

import PyPDF2
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import AzureOpenAI
from langchain.chains.question_answering import load_qa_chain
# document process
if __name__ == "__main__":
    print("hi kamal")
    pdf_path = "D:/project/2023/in/MyResume.pdf"
    loader = PyPDFLoader(file_path=pdf_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=30, separator="\n"
    )
    docs = text_splitter.split_documents(documents=documents)
    print(len(docs))
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        deployment="embedding"
    )
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("D:/project/2023/in/faiss/","faiss_index_react")
    new_vectorstore = FAISS.load_local("D:/project/2023/in/faiss/", embeddings,"faiss_index_react")
    qa = RetrievalQA.from_chain_type(
        llm=AzureOpenAI(), chain_type="stuff", retriever=new_vectorstore.as_retriever()
    )
    res = qa.run("is  he suitable for darabricks Developer? then how   ")
"""

Retrying langchain.llms.openai.completion_with_retry.<locals>._completion_with_retry in 4.0 seconds as it raised APIError: Invalid response object from API: '{ "statusCode": 404, "message": "Resource not found" }' (HTTP response code was 404).


"""