from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
import json
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    OPENAI_API_KEY: str
    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY

# Create instance of FastAPI
app = FastAPI()

# Schema for retrieving documents
class QueryRequest(BaseModel):
    query: str
    num_results: int = 3


# Initialize ChromaDB with persistence
CHROMA_DIR = "./chroma_db"
embedding_model = OpenAIEmbeddings()
vector_store = Chroma(collection_name="chatbot",embedding_function=embedding_model,persist_directory=CHROMA_DIR)
llm = ChatOpenAI(model="gpt-4o-mini")

@app.post("/upload/")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document and store its embeddings in ChromaDB.
    """
    try:
        # Read the file content
        loader = TextLoader(file.filename)
        documents = loader.load()

        #split the documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=200,length_function=len)
        splits = text_splitter.split_documents(documents)

        #store in chromaDB
        vector_store.add_documents(splits)

        return {"message": "Document uploaded successfully", "filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding document: {str(e)}")

@app.post("/chat/")
async def query_documents(request:QueryRequest):
    """
    Give the results to llm and get answer.
    """
    try:
        results = vector_store.similarity_search(request.query,k=request.num_results)
        context = "\n".join([result.page_content for result in results])

        # Define prompt for question-answering
        instructions = (
            "You are an assistant that answers questions based strictly on the provided context. "
            "Provide the answer in a clean, conversational, and user-friendly format. Avoid using any special characters, bullet points, or unnecessary formatting. "
            "If the answer is not in the context, say 'I don't know.'"
        )
        full_prompt = f"""{instructions}
        Context:{context}
        Question: {request.query}"""

        # Step 4: Generate response using LLM
        model_response = llm.invoke(full_prompt)

        # store the converstion in a json file
        data = {
            "question": request.query,
            "answer": model_response.content
        }

        with open("qa.jsonl","a") as f:
            json.dump(data,f)
            f.write("\n")

        return {
            "query": request.query,
            "model_response": model_response.content
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat request: {str(e)}")