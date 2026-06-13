from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage,SystemMessage, AIMessage
from langchain_ollama import ChatOllama ,OllamaEmbeddings 

load_dotenv()

#connecting to my document db
persistent_directory = "db/chroma_db"
embeddings = OllamaEmbeddings(model ="nomic-embed-text", base_url="http://localhost:11434")
db=Chroma(persist_directory=persistent_directory,embedding_function=embeddings)

#setting up ai model
model=ChatOllama(model = "llama3.2")

#storing conversation as message
chat_history = []

def ask_question(user_question): 
    print(f"\n Your question : {user_question}")

    #making question clear using history
    if chat_history:
        #if there exists chat history we will ask ai to rewrite the question
        messages = (
        [SystemMessage(content="Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question.")]
        + chat_history
        + [HumanMessage(content=f"New question: {user_question}")]
    )
        
        result = model.invoke(messages)
        search_question = result.content.strip()

    else: #this when the chat history is empty
        search_question=user_question
    
    #Find elevant documents
    retriever =db.as_retriever(search_kwargs={"k":3})
    docs =retriever.invoke(search_question)

    #creating final prompt
    combined_input= f"""Based on the following documents, please answer this question :{user_question}
    
    Documents:
    {"\n".join([f"{doc.page_content}" for doc in docs])}

    Please provide clear and helpful answer but only use the information from these documents. IF you cannot dins the answer then say "I dont have enought information to answer the questions"
    """

    messages = [
        SystemMessage(content="You are a helpful assistant that answers questions based on provided documents and conversation history"), 
    ]+ chat_history + [
        HumanMessage(content=combined_input), #the actual message that llm should follow
    ]

    #sending request to llm
    result =model.invoke(messages)
    answer = result.content

    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))

    print("Answer: " , answer)
    return answer

#chat loop
def start_Chat():
    print("Ask me a question. Type 'Quit' to exit")

    while True:
        question = input("your question:")

        if question.lower() =='quit':
            print("Byebyeee")
            break

        ask_question(question)

if __name__ =="__main__":
    start_Chat()