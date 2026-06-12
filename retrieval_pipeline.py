from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from dotenv import load_dotenv

#our embeddings of the docs , the vector db we want access to its data
persistent_directory ="db/chroma_db"

#initializing the embedding model and we have to use the same embediing model which was used before vector db
embedding_model= OllamaEmbeddings( model="nomic-embed-text", base_url="http://localhost:11434")

#recreating the vector store, pointing the new vector store with our old particular data
db =Chroma.from_documents(
        embedding=embedding_model, #we have specified the model above
        persistent_directory=persistent_directory, #this is where we store it locally
        collection_metadata={"hnsw:space":"cosine"} #the algorithm is cosine similarity to do the matching
    )

#using the instance of db to create retriever component
#this retriever is going to retrieve the top 3 chunks of higher similarity with query embeddings
retriever=db.as_retriever(search_kwargs={"k":3} )

relevant_docs=retriever.invoke(query)