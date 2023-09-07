import os

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

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

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("faiss_index_react")

    new_vectorstore = FAISS.load_local("venv/faiss_index_react", embeddings)
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(), chain_type="stuff", retriever=new_vectorstore.as_retriever()
    )
    # res = qa.run("Give me the intro 500 words ")
    # res = qa.run("is  he suitable for data engineering job?  ")
    res = qa.run("is  he suitable for darabricks Developer? then how   ")
    # res = qa.run("is  he suitable for web Developer?  ")
    # res = qa.run("what are the missing skils for  data engineering job?  ")
    print(res)
