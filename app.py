# import required libraries
import streamlit as st
from ingest import ingest_pdf
from retriever import ask_question

#title
st.title("Chat With PDF", text_alignment="center")
if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


#upload pdf
with st.sidebar:
    upload_file = st.file_uploader("Upload PDF", type="pdf")
    if upload_file is not None:
        with open("temp.pdf", "wb") as f:
            f.write(upload_file.read())
        ingest_pdf("temp.pdf")
        st.success("PDF processesd")

# question 
question = st.chat_input("Ask a question....")
if question:
    st.session_state.messages.append({'role':'user', 'content':question})
    response = ask_question(question)
    st.session_state.messages.append({'role':'assistant', 'content':response})
    st.rerun()