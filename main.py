from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Create instance of FastAPI
app = FastAPI()

# Schema for retrieving documents
class QueryRequest(BaseModel):
    query: str
    num_results: int = 3

# Initialize ChromaDB with persistence
CHROMA_DIR = "./chroma_db"
embedding_model = OpenAIEmbeddings()
vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding_model)

@app.post("/upload/")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document and store its embeddings in ChromaDB.
    """
    try:
        # Read the file content
        content = await file.read()
        content_text = content.decode("utf-8")
        document = Document(page_content=content_text)

        # Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
        all_splits = text_splitter.split_documents([document])
        split_texts = [split.page_content for split in all_splits]

        # Generate unique IDs for each chunk
        chunk_ids = [f"{file.filename}_{i}" for i in range(len(split_texts))]

        # Add chunks to ChromaDB
        vectorstore.add_texts(
            texts=split_texts,
            metadatas=[{"filename": file.filename}] * len(split_texts),
            ids=chunk_ids
        )

        return {"message": "Document uploaded successfully", "filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding document: {str(e)}")

@app.get("/documents/")
async def list_documents():
    """
    List all the document IDs stored in ChromaDB.
    """
    try:
        collections = vectorstore.get()
        document_ids = collections.get("ids", [])
        return {"documents": document_ids}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")

@app.post("/query/")
async def query_documents(request: QueryRequest):
    """
    Query documents by providing a search string.
    """
    try:
        results = vectorstore.similarity_search(request.query, k=request.num_results)
        return {
            "query": request.query,
            "results": [
                {
                    "id": result.metadata.get("filename"),
                    "content": result.page_content
                }
                for result in results
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying documents: {str(e)}")
