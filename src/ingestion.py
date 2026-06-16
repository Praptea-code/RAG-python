import os

#these will help us to read any ppt,pdf,text file from particular directory
from langchain_community.document_loaders import TextLoader, DirectoryLoader
#this helps to chunk it up after loading
from langchain_text_splitters import CharacterTextSplitter
#to convert chunks to vector embeddings
from langchain_community.embeddings import OllamaEmbeddings
#vector db to store embeddings, and chroma db can be hosted locally so easier
from langchain_chroma import Chroma
#to load environment variables
from dotenv import load_dotenv

load_dotenv()

#loading the document
def load_documents(docs_path="../docs"):
    #loads all text files from docs directory
    print(f"Loading all documents from the {docs_path}")

    #checking if directory exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError("The file does not exixst")
    
    #load all .txt files from the directory docs
    loader= DirectoryLoader(
        path=docs_path,
        glob="*.txt", #only look for txt files
        loader_cls=TextLoader, #using txt loader class we imported this before 
        loader_kwargs={"encoding": "utf-8"}
        )

    documents=loader.load() # this will list langchain documents which have page content, attributes and metadata arttibutes

    if len(documents)==0:
        raise FileNotFoundError("No .txt files found add your documents first")
    
    # for i, doc in enumerate(documents[:2]):
    #     print(f"\nDocument {i+1}:")
    #     print(f" Source:{doc.metadata['source']}")
    #     print(f"Content length: {len(doc.page_content)} characters")
    #     print(f"Content preview: {doc.page_content[:100]}")
    #     print(f"metadata:{doc.metadata}")
    
    return documents 

#chunking part
def split_documents(documents):

    #the class charactertextsplitter is one the most basic text splitting class in langchain
    text_splitter =CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0
    )

    chunks= text_splitter.split_documents(documents)

    # if chunks:
    #     for i, chunk in enumerate(chunks[:5]):
    #         print(f"\n Chunk {i+1}")
    #         print(f"Source: {chunk.metadata['source']}")
    #         print(f"Length: {len(chunk.page_content)} characters")
    #         print("content:")
    #         print(chunk.page_content)
            
    #         if len(chunks) >5:
    #             print(f"{len(chunks)-5 }, more chunks")
    
    return chunks

def create_vector_store(chunks, persist_directory="db/chroma_db"):
    #embedding model to convert the chunks in vector embeddings
    #initializing the embedding model, running locally via ollama
    embedding_model = OllamaEmbeddings(
        model="nomic-embed-text", 
        base_url="http://localhost:11434"
    )

    print(f"Embedding {len(chunks)} chunks...")

    #creating chromadb vector store without inserting documents yet
    #we initialize it empty first so we can add chunks in batches
    vectorstore = Chroma(
        embedding_function=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )

    #sending 50 chunks at a time instead of 1, much faster
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        vectorstore.add_documents(batch)
        print(f"  Embedded {min(i+batch_size, len(chunks))}/{len(chunks)} chunks")

    print("Finished creating the store")
    print(f"Created and saved to {persist_directory}")

    return vectorstore

def main():
    print("Main function")

    #loading the files
    documents=load_documents(docs_path="../docs")

    #chunking the documents
    chunks=split_documents(documents) 

    #embedding and storing in vector db
    vectorstore=create_vector_store(chunks)

if __name__ == "__main__":
    main()

