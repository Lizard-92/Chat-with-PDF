## import required libraries
from dotenv import load_dotenv
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import shutil
import os

# loadenv
load_dotenv()

def ingest_pdf(pdf_file):
    #load pdf files
    #if os.path.exists("./nike_db"):
        #shutil.rmtree("./nike_db")


    loader = PDFPlumberLoader(pdf_file)
    print(f"File loading....")
    pages = loader.load()
    print(f"PDF loaded successfully! File Length {len(pages)}")

    #splitting pdf file in chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=180, separators=["\n\n", "\n", " "])
    print(f"Split Documents in proces...")
    chunks = splitter.split_documents(pages) 
    print(f"Chunks complete {len(chunks)}")

    #embedding model
    print(f"Initilizaing Model....")
    embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5')
   
    #embedding
    vectorestore = Chroma.from_documents(chunks, embeddings, persist_directory="./nike_db")
    print(f"Embedding in process...")
    print("Saved to ChromaDB!")

if __name__ == "__main__":
    ingest_pdf("nke-10k-2023.pdf")




