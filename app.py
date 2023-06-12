import os

from flask import Flask,request,jsonify

from werkzeug.utils import secure_filename
from werkzeug.exceptions import HTTPException

import logging
from logging.handlers import RotatingFileHandler

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import ChatVectorDBChain, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader

from constants.upload import UPLOAD_FOLDER
from constants.persists import PERSIST_DIRECTORY


os.environ["OPENAI_API_KEY"] = 'sk-YOUR_KEY'


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Set up logging
handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=1)
handler.setLevel(logging.INFO)
app.logger.addHandler(handler)


def generate_davinci_blog_post(vectordb, topic, temperature, k):
    prompt_template = """Use the context below to write a 400 word blog post about the topic below:
            Context: {context}
            Topic: {topic}
            Blog post:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "topic"]
    )
    llm = OpenAI(temperature=temperature) #don't get too crazy with this
    chain = LLMChain(llm=llm, prompt=PROMPT)
    
    docs = vectordb.similarity_search(topic, k=k)
    inputs = [{"context": doc.page_content, "topic": topic} for doc in docs]
    return chain.apply(inputs)

def generate_gpt4_blog_post(vectordb, topic, temperature, k):
    all_questions=[]
    all_answers=[]

    docs = vectordb.similarity_search(topic, k=k)

    for doc in docs:
        context = doc.page_content
        all_questions.append(
            f"""
            Use the context below to write a 400 word blog post about the topic below:
            Context: {context}
            Topic: {topic}
            Blog post:
            """
        )
    # was OpenAI => API changes
    chain = ChatVectorDBChain.from_llm(ChatOpenAI(temperature=temperature, model_name="gpt-4"), vectordb)

    for question in all_questions:
        answer = chain.run(question=question, chat_history=[])
        all_answers.append(answer)

    return all_answers

def save_embeddings(embeddings,pages,collection_name):
    vectordb = Chroma.from_documents(   
        documents=pages,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory= PERSIST_DIRECTORY
    )   
    vectordb.persist()

def retrieve_embeddings(collection_name,embeddings):
    vectordb = Chroma(
        collection_name=collection_name,
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings
    )
    # print("Embeddings collection: ",vectordb._collection)
    return vectordb


@app.errorhandler(Exception)
def handle_exception(e):
    # Handle HTTPException separately
    if isinstance(e, HTTPException):
        return jsonify(error=str(e)), e.code
    app.logger.error(f"An error occurred: {e}")
    return jsonify(error="Internal Server Error"), 500

@app.before_first_request
def create_folders():
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    if not os.path.exists('persits'):
        os.makedirs('persits')

@app.route("/")
def hello_world():
    return "Hello wrold;"


@app.route("/upload", methods=["POST"])
def upload_pdf():
    """
    It saves the passed pdf file in the uploads directory
    """

    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400
    if file:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))


    return "File has been uploaded;"


@app.route("/embed", methods=["POST"])
def embed_pdf():
    """
    It creates the document's embeddings, stored in the perists direcotry
    """

    filename = request.json['filename']
    # chunk it & save embeddings

    loader = PyPDFLoader(f'{UPLOAD_FOLDER}/{filename}')
    pages = loader.load_and_split()

    embeddings = OpenAIEmbeddings()
    
    try:
        collection_name = request.json['collection_name']

        save_embeddings(embeddings=embeddings,pages=pages,collection_name=collection_name)
    except KeyError:
        print("Saving embeddings to default chroma collection name")
        save_embeddings(embeddings=embeddings,pages=pages,collection_name="test_collection")

    return "Embeddings created;"


@app.route("/create", methods=["POST"])
def create_paragraph():
    """
    Creates the actual blog post
    You need to pass the topic, the model to be used and a name for the database collection
    """
    topic = request.json['topic']
    model = request.json['model']

    try:
        collection_name = request.json['collection_name']

        embeddings = OpenAIEmbeddings()
        vectordb = retrieve_embeddings(collection_name=collection_name,embeddings=embeddings)
    except KeyError:
        print("Using the default chroma collection name;")
        embeddings = OpenAIEmbeddings()
        vectordb = retrieve_embeddings(collection_name="test_collection",embeddings=embeddings)
        

    # OpenAI has 2 endpoints that are relevant for our use case : Completions & ChatCompletions
    # the Completions API does not have access to the GPT4 Model - it uses davinci => much cheaper & faster; GPT3.5 comparable performance
    # the ChatCompletions API does have access to GPT4 - hence I created a workaround to get the same output as a regular Competions request 
    if model=='davinci':
        return generate_davinci_blog_post(vectordb=vectordb,topic=topic, temperature=0.3,k=4)
    if model=='gpt4':
        return generate_gpt4_blog_post(topic=topic, vectordb=vectordb, temperature=0.3, k=2) #better model=> takes longer to compute
    return "Please choose a valid model;"


