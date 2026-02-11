from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai  import ChatOpenAI
from langchain_classic.chains import RetrievalQA

docs = [
    "Refunds are processed within 7 days after approval.",
    "Users can reset their password using the forgot password link.",
    "Customer support is available 24/7 via email."
]

embeddings = OpenAIEmbeddings() # embeddings is used to convert text to vector
db = FAISS.from_texts(docs, embeddings) # FAISS is a vector database

llm = ChatOpenAI(model="gpt-4.1")

qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever()) # This is essentially where RAG magic happens it is basically RAG Only takes vectorDb, llm and query later basically use to build the context

result = qa.run("How long does refund take?")
print(result)
