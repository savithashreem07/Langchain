from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_community.document_loaders import TextLoader
from langchain.chains import RetrievalQA,  ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
import panel as pn
import param
import os
import warnings
from openai import OpenAI

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


def load_db(file, chain_type, k):
    # load documents
    loader = PyPDFLoader(file)
    documents = loader.load()
    # split documents
    text_splitter = RecursiveCharacterTextSplitter(
           chunk_size=1000, 
           chunk_overlap=150)
    docs1 = text_splitter.split_documents(documents)
    # define embedding
    embeddings = OpenAIEmbeddings()
    # create vector database from data
    db = DocArrayInMemorySearch.from_documents(docs1, 
           embeddings)
    # define retriever
    retriever = db.as_retriever(search_type="similarity", 
           search_kwargs={"k": k})
    # create a chatbot chain. Memory is managed externally.
    qa = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0), 
        chain_type=chain_type, 
        retriever=retriever, 
        return_source_documents=True,
        return_generated_question=True,
    )
    return qa 

class cbfs(param.Parameterized):
    chat_history = param.List([])
    answer = param.String("")
    db_query  = param.String("")
    db_response = param.List([])

    def __init__(self,  **params):
        super(cbfs, self).__init__( **params)
        self.panels = []
        self.loaded_file = "docs/MachineLearning-Lecture01.pdf"
        self.qa = load_db(self.loaded_file,"stuff", 4)

    def call_load_db(self, count):
        # init or no file specified :
        if count == 0 or file_input.value is None:  
            return pn.pane.Markdown(f"Loaded File: {self.loaded_file}")
        else:
            file_input.save("temp.pdf")  # local copy
            self.loaded_file = file_input.filename
            button_load.button_style="outline"
            self.qa = load_db("temp.pdf", "stuff", 4)
            button_load.button_style="solid"
        self.clr_history()
        return pn.pane.Markdown(f"Loaded File: {self.loaded_file}")

    def convchain(self, query):
        if not query:
            return pn.WidgetBox(pn.Row('User:', 
               pn.pane.Markdown("", width=600)), scroll=True)
        result = self.qa({"question": query, 
                          "chat_history": self.chat_history})
        self.chat_history.extend([(query, result["answer"])])
        self.db_query = result["generated_question"]
        self.db_response = result["source_documents"]
        self.answer = result['answer'] 
        self.panels.extend([
            pn.Row('User:', pn.pane.Markdown(query, width=600)),
            pn.Row('ChatBot:', pn.pane.Markdown(self.answer, 
               width=600, 
               style={'background-color': '#F6F6F6'}))
        ])
        inp.value = ''  # clears loading indicator when cleared
        return pn.WidgetBox(*self.panels, scroll=True)
    
    @param.depends('db_query')
    def display_db_query(self):
        if not self.db_query:
            return pn.Column(
                pn.Row(pn.pane.Markdown(f"Last question to DB:", 
            styles={'background-color': '#F6F6F6'})),
                pn.Row(pn.pane.Str("no DB accesses so far"))
            )
        return pn.Column(pn.Row(pn.pane.Markdown(f"DB query:", styles={'background-color': '#F6F6F6'})), pn.pane.Str(self.db_query))

    # @param.depends('db_query ', )
    # def convchain(self):
    #     if not self.db_query :
    #         return pn.Column(
    #             pn.Row(pn.pane.Markdown(f"Last question to DB:", 
    #         styles={'background-color': '#F6F6F6'})),
    #             pn.Row(pn.pane.Str("no DB accesses so far"))
    #         )
    #     return pn.Column(pn.Row(pn.pane.Markdown(f"DB query:", styles={'background-color': '#F6F6F6'})),pn.pane.Str(self.db_query ))
    
    @param.depends('db_response', )
    def get_sources(self):
        if not self.db_response:
            return pn.WidgetBox(pn.Row(pn.pane.Str("No DB response yet")), width=600, scroll=True)
        rlist=[pn.Row(pn.pane.Markdown(f"Result of DB lookup:", 
            styles={'background-color': '#F6F6F6'}))]
        for doc in self.db_response:
            rlist.append(pn.Row(pn.pane.Str(doc)))
        return pn.WidgetBox(*rlist, width=600, scroll=True)

    @param.depends('convchain', 'clr_history') 
    def get_chats(self):
        if not self.chat_history:
            return pn.WidgetBox(
                  pn.Row(pn.pane.Str("No History Yet")), 
                   width=600, scroll=True)
        rlist=[pn.Row(pn.pane.Markdown(
            f"Current Chat History variable", 
            styles={'background-color': '#F6F6F6'}))]
        for exchange in self.chat_history:
            rlist.append(pn.Row(pn.pane.Str(exchange)))
        return pn.WidgetBox(*rlist, width=600, scroll=True)

    def clr_history(self,count=0):
            self.chat_history = []
            return
    
chatbot = cbfs()
# class ChatApp(param.Parameterized):
#     loaded_file = "docs/MachineLearning-Lecture01.pdf"
#     chatbot = cbfs()

@app.route('/submit', methods=['POST'])
def submit():
    language = 'en'
    answer = ''
    question = ''

    if request.method == 'POST':
        query = request.form['query']
        chatbot.convchain(query)
        return render_template('index.html', answer=chatbot.answer, db_query=chatbot.db_query, db_response=chatbot.db_response)
    
@app.route('/')
def index():
    # template = pn.template.FastListTemplate(
    #     site="Chatbot UI",
    #     title="Chatbot Interface",
    #     header_background="#F6F6F6",
    #     main=[pn.panel(chatbot.param, show_name=False)],
    #     main_max_width="768px",
    # )
    # template.theme(pn.template.DefaultTheme)
    # return template.servable()
    return render_template('index.html', answer="", db_query="", db_response=[])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
