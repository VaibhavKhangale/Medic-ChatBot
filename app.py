from flask import Flask, render_template, jsonify, request
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY=os.getenv("GEMINI_API_KEY")

os.environ["PINECONE_API_KEY"]=PINECONE_API_KEY
os.environ["GEMINI_API_KEY"]=GEMINI_API_KEY

index_name="medic-bot"
embeddings=download_embeddings()

docsearch=PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever=docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

chat_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=GEMINI_API_KEY,
    convert_system_message_to_human=True
)

prompt=ChatPromptTemplate.from_messages(
    [
        ("system",system_prompt),
        ("human","{input}"),
    ]
)

qna_chain=create_stuff_documents_chain(chat_model, prompt)
rag_chain=create_retrieval_chain(retriever, qna_chain)

@app.route("/get", methods=["GET","POST"])
def chat():
    msg=request.form["msg"]
    input=msg
    print(input)
    response=rag_chain.invoke({"input":msg})
    print("Response : ", response["answer"])
    return str(response["answer"])

@app.route("/")
def index():
    return render_template('chat.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)