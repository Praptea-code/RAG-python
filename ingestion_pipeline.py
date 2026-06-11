import os

#these will help us to read any ppt,pdf,text file from particular directory
from langchain_community.document_loaders import TextLoader, DirectoryLoader
#this helps to chunk it up after loading
from langchain_text_splitters import CharacterTextSplitter
#to convert chunks to vector embeddings
from langchain_openai import OpenAIEmbeddings
#vector db to store embeddings, and chroma db can be hosted locally so easier
from langchain_chroma import Chroma
#to load environment variables
from dotenv import load_dotenv

load_dotenv()

def load_documents(docs_path="docs"):
    #loads all text files from docs directory
    print(f"Loading all documents from the {docs_path}")

    #checking if directory exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError("The file does not exixst")
    
    #load all .txt files from the directory docs
    loader= DirectoryLoader(
        path=docs_path,
        glob="*.txt", #only look for txt files
        loader_cls=TextLoader #using txt loader class we imported this before 
        )

    documents=loader.load() # this will list langchain documents which have page content, attributes and metadata arttibutes

    if len(documents)==0:
        raise FileNotFoundError("No .txt files found add your documents first")
    
    for i, doc in enumerate(documents[:2]):
        print(f"\nDocument {i+1}:")
        print(f" Source:{doc.metadata['source']}")
        print(f"Content length: {len(doc.page_content)} characters")
        print(f"Content preview: {doc.page_content[:100]}")
        print(f"metadata:{doc.metadata}")
    
    return documents 

def main():
    print("Main function")

    #loading the files
    documents=load_documents(docs_path="docs")

if __name__ == "__main__":
    main()

