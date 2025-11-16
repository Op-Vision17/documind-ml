import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Validate required environment variables
required_vars = [
    'PINECONE_API_KEY',
    'PINECONE_INDEX',
    'NODE_SERVICE_TOKEN',
    'NODE_URL',
    'GROQ_API_KEY'
]

missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    print(f"⚠️  Missing required environment variables: {', '.join(missing_vars)}")
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Node service configuration
NODE_URL = os.getenv("NODE_URL", "http://localhost:5000")
NODE_SERVICE_TOKEN = os.getenv("NODE_SERVICE_TOKEN")

# ML service configuration
ML_HOST = os.getenv("ML_HOST", "0.0.0.0")
ML_PORT = int(os.getenv("ML_PORT", 8000))

# Embedding configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "none")

# RAG configuration
TOP_K = int(os.getenv("TOP_K", 4))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))

# Pinecone configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-west1-gcp")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "documind-index")

# Groq/LLM configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Print loaded config (without sensitive data)
print("✅ Configuration loaded:")
print(f"   - NODE_URL: {NODE_URL}")
print(f"   - PINECONE_INDEX: {PINECONE_INDEX}")
print(f"   - EMBEDDING_MODEL: {EMBEDDING_MODEL}")
print(f"   - ML_PORT: {ML_PORT}")
print(f"   - GROQ_API_KEY: {'***SET***' if GROQ_API_KEY else 'MISSING'}")