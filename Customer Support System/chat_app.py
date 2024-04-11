import os
import warnings
from openai import OpenAI
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain

from flask import Flask, redirect, render_template, request

app = Flask(__name__)

# Define OpenAI API_KEY
with open("/home/savitha07/.env") as env:
    for line in env:
        key, value = line.strip().split('=')
        os.environ[key] = value

client = OpenAI(
    api_key=os.environ.get('OPENAI_API_KEY'),
)

os.environ["TAVILY_API_KEY"] = os.environ.get('OPENAI_API_KEY')

warnings.filterwarnings("ignore")

delimiter = "####"

# Load the PDF file
loader = PyPDFLoader("docs/2023Catalog.pdf")
documents = loader.load()

# split documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
docs1 = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
vectordb = DocArrayInMemorySearch.from_documents(docs1, embeddings)

memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    input_key='question',
    output_key='answer',
    return_messages=True,
    k=5
)

def get_completion_from_messages(messages, 
                                 model="gpt-3.5-turbo", 
                                 temperature=0, 
                                 max_tokens=500):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature, 
        max_tokens=max_tokens, 
    )
    return response.choices[0].message.content

def get_translation(text, language):
    system_message = f"""Assume you are a professional translator.\
    The text will be delimited by tripe backticks.\
    {delimiter}{text}{delimiter}"""

    user_message = f"""Translate the text into {language} language."""

    messages =  [  
    {'role':'system', 
    'content': system_message},    
    {'role':'user', 
    'content': f"{delimiter}{user_message}{delimiter}"},  
    ]

    translate = get_completion_from_messages(messages)
    print(translate + "\n")
    return translate

def generate_response(memory, question):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    template = """
    Assume you are San Francisco Bay University Assistant. Your task it to answer questions about the catalog. Use the following pieces of \
    context to answer the question at the end. \
    If you don't know the answer, \
    just say that you don't know, don't try \
    to make up an answer.
    {context}
    Question: {question}
    Helpful Answer:"""

    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=vectordb.as_retriever(),
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT},
        get_chat_history=lambda h : h,
        memory=memory,
    )

    return qa_chain.invoke(input={"question": question})


@app.route("/", methods=["GET", "POST"])
def index():
    result = ''
    language = request.form.get('language')
    question = request.form.get('question')
    translate = request.form.get('translate') == True

    result = generate_response(memory, question)['answer']

    if translate:
        result = get_translation(result, language)

    print(result)

    return render_template('index.html', result = result, language = language)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
