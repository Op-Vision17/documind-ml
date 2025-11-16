# Import config first to load environment variables
from .config import ML_HOST, ML_PORT

from fastapi import FastAPI
from pydantic import BaseModel
from .rag_engine import process_ingest, process_answer

app = FastAPI(
    title="DocuMind ML Service",
    version="1.0",
)

# ---------- MODELS ----------
class IngestBody(BaseModel):
    fileId: str
    fileUrl: str
    originalName: str

class QueryBody(BaseModel):
    query: str
    top_k: int = 4  # Optional parameter with default


# ---------- ROUTES ----------

@app.post("/ingest")
def ingest(data: IngestBody):
    """
    Ingest a PDF located at fileUrl:
    - Download from Supabase
    - Extract text
    - Chunk
    - Embed
    - Store in Pinecone
    - Notify Node backend
    """
    result = process_ingest(
        data.fileId,
        data.fileUrl,
        data.originalName
    )
    return result


@app.post("/answer")
def answer(data: QueryBody):
    """
    Search Pinecone using query embeddings
    Return combined context as simplified answer
    """
    return process_answer(data.query, data.top_k)


@app.get("/")
def home():
    return {"message": "DocuMind ML service running ✔"}


@app.get("/health")
def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "DocuMind ML Service",
        "version": "1.0"
    }