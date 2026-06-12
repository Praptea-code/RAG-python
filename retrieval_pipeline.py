from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage,SystemMessage

#our embeddings of the docs , the vector db we want access to its data
persistent_directory ="db/chroma_db"

#initializing the embedding model and we have to use the same embediing model which was used before vector db
embedding_model= OllamaEmbeddings( model="nomic-embed-text", base_url="http://localhost:11434")

#recreating the vector store, pointing the new vector store with our old particular data
db =Chroma(
        persist_directory=persistent_directory, #this is where we store it locally
        embedding_function=embedding_model, #we have specified the model above
        collection_metadata={"hnsw:space":"cosine"} #the algorithm is cosine similarity to do the matching
    )

#using the instance of db to create retriever component
#this retriever is going to retrieve the top 5 chunks of higher similarity with query embeddings
retriever=db.as_retriever(search_kwargs={"k":5} )

query="What was the original name of Microsoft before it became Microsoft?"

#calling the retriever
relevant_docs=retriever.invoke(query)

print(f"User Query :{query}")

print("Context")
for i, doc in enumerate(relevant_docs,1):
    print(f"Document {i}: \n {doc.page_content} \n")

#this is the prompt prepared to send to the llm model
#we pass the query and take every document and extract its text
combined_input= f"""Based on the following documents, please answer this question :{query} 

Documents:
{chr(10).join([f"{doc.page_content}" for doc in relevant_docs])}
Please provide clear and helpful answer but only use the information from these documents. IF you cannot dins the answer then say "I dont have enought information to answer the questions"
"""
#this is our llm model we load the llm model
model= ChatOllama(model="llama3.2")

messages = [
    SystemMessage(content="You are a helpful assistant"), #the system is setting the behavior, rules as modern chat models do not take  single plain prompt anymore
    HumanMessage(content=combined_input), #the actual message that llm should follow
]

#sending request to llm
result =model.invoke(messages)

print("Generated Response")
print("Content only:")
#ai message sends content we are just taking the text from it 
print(result.content)