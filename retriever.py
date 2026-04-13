from dotenv import load_dotenv
import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory, RunnablePassthrough
from langchain_community.chat_message_histories import ChatMessageHistory
from operator import itemgetter

# load env
load_dotenv()

#load embedding model from huggingface
embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5')

# retriving vector store
vector_store = Chroma(persist_directory="./nike_db", embedding_function=embeddings)

#defining retriever
retriever = vector_store.as_retriever()

# llm model
llm = ChatGroq(
    model='llama-3.3-70b-versatile',
    temperature=0.1
)

template = ChatPromptTemplate.from_messages([
    ("system", "You are helpful assistant. answer only from context. if answer not in context say 'I do not know'. give proper answer do not truncate.  Use only this context to answer:\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

#combining retriever and question
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

#chain = (
    #{"context": retriever | format_docs, "question": RunnablePassthrough()}
    #| template
    #| llm  # Must align with | operators above (same indent level)
#)
#print(chain.invoke("Who is Executive Chairman?"))

chain = (
    RunnablePassthrough.assign(context=itemgetter("question") | retriever | format_docs)
    | template
    | llm
)

#store for messages
store = {}
def get_session_history(session_id):
   if session_id not in store:
        store[session_id] = ChatMessageHistory()
   return store[session_id]

#combined everything together
bot = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="chat_history")

def ask_question(question, session_id="abc123"):
    response = bot.invoke(
        {"question":question},
        config={"configurable":{"session_id":session_id}}
    )
    return response.content

if __name__ == "__main__":
    print(ask_question("Who is Executive Vice President and Chief Financial Officer?"))


    