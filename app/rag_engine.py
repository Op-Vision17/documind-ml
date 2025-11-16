import os
import tempfile
import requests
from groq import Groq
from pinecone import Pinecone
from .pdf_utils import extract_pdf_text
from .chunker import chunk_text
from .utils import notify_node_update

# Import configuration (this loads .env first)
from .config import (
    PINECONE_API_KEY,
    PINECONE_INDEX,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    GROQ_API_KEY
)

# -----------------------------
# INITIALIZE GROQ CLIENT
# -----------------------------
client = Groq(api_key=GROQ_API_KEY)

# -----------------------------
# LAZY LOAD EMBEDDER (Saves ~200MB memory at startup)
# -----------------------------
_embedder = None

def get_embedder():
    """Lazy load embedder to save memory on startup"""
    global _embedder
    if _embedder is None:
        print("🔄 Loading embedding model (this may take a moment)...")
        from .embedder import Embedder
        _embedder = Embedder(EMBEDDING_MODEL)
        print("✅ Embedding model loaded successfully")
    return _embedder

# -----------------------------
# INITIALIZE PINECONE CLIENT
# -----------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)

# Check index exists (don't create at startup to save time/memory)
try:
    index_names = [idx["name"] for idx in pc.list_indexes()]
    if PINECONE_INDEX in index_names:
        print(f"✅ Index {PINECONE_INDEX} already exists")
    else:
        print(f"⚠️ Index {PINECONE_INDEX} not found. It will be created on first use.")
except Exception as e:
    print(f"⚠️ Could not check Pinecone indexes: {e}")

index = pc.Index(PINECONE_INDEX)

# ------------------------------------------------------
# DOWNLOAD PDF FROM SUPABASE SIGNED URL
# ------------------------------------------------------
def download_file(url):
    r = requests.get(url, stream=True)
    r.raise_for_status()

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    with open(tmp.name, "wb") as f:
        for chunk in r.iter_content(8192):
            if chunk:
                f.write(chunk)

    return tmp.name

# ------------------------------------------------------
# INGEST FLOW — CLOUD READY
# ------------------------------------------------------
def process_ingest(fileId, fileUrl, originalName):
    try:
        # 1. Download from Supabase signed URL
        local_path = download_file(fileUrl)

        # 2. Extract raw text
        text = extract_pdf_text(local_path)
        if not text.strip():
            notify_node_update(fileId, "failed", "Empty text extracted")
            return {"ok": False, "reason": "Empty text"}

        # 3. Chunk text
        chunks = chunk_text(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

        # 4. Embed chunks (lazy load embedder here)
        embedder = get_embedder()
        vectors = embedder.embed_documents(chunks)

        # 5. Upsert to Pinecone
        batch = []
        for i, vec in enumerate(vectors):
            vector = vec.tolist()
            meta = {
                "fileId": fileId,
                "chunkId": i,
                "text": chunks[i],
                "source": originalName
            }

            batch.append((
                f"{fileId}_{i}",   # vector ID
                vector,
                meta
            ))

        # Batch upload (100 per batch)
        batch_size = 100
        for i in range(0, len(batch), batch_size):
            index.upsert(vectors=batch[i:i + batch_size])

        # 6. Notify Node backend
        notify_node_update(fileId, "indexed", pages=len(chunks))

        # Cleanup
        os.remove(local_path)

        return {"ok": True, "indexed_chunks": len(chunks)}

    except Exception as e:
        notify_node_update(fileId, "failed", str(e))
        return {"ok": False, "reason": str(e)}

# ------------------------------------------------------
# ANSWER ENDPOINT — RAG QUERY
# ------------------------------------------------------
def process_answer(query, top_k=4):
    """Process a query and return an AI-generated answer with sources"""
    try:
        print(f"🔍 Processing query: {query}")
        
        # 1. Embed query (lazy load embedder here)
        embedder = get_embedder()
        qvec = embedder.embed_query(query).tolist()

        # 2. Search Pinecone
        results = index.query(
            vector=qvec,
            top_k=top_k,
            include_metadata=True
        )

        matches = results.get("matches", [])
        
        if not matches:
            return {
                "answer": "I couldn't find any relevant information in your documents to answer this question. Please make sure documents are uploaded and indexed.",
                "sources": []
            }
        
        print(f"✅ Found {len(matches)} relevant chunks")
        
        contexts = []
        sources = []

        for m in matches:
            meta = m.get("metadata", {})
            score = m.get("score", 0)
            
            # Only include high-quality matches (score > 0.3)
            if score > 0.3:
                contexts.append(meta.get("text", ""))
                sources.append({
                    "fileId": meta.get("fileId"),
                    "chunkId": meta.get("chunkId"),
                    "score": round(score, 4),
                    "source": meta.get("source", "Unknown")
                })

        if not contexts:
            return {
                "answer": "The retrieved documents don't seem relevant enough to answer your question confidently.",
                "sources": sources
            }

        # Combine contexts (limit to avoid token limits)
        combined_context = "\n\n---\n\n".join(contexts[:3])  # Use top 3 chunks only
        
        print(f"📝 Generating answer with Groq...")

        # 3. Generate answer using LLM
        final_answer = generate_llm_answer(combined_context, query)
        
        print(f"✅ Answer generated successfully")

        return {
            "answer": final_answer,
            "sources": sources,
            "chunks_used": len(contexts)
        }

    except Exception as e:
        print(f"❌ Error in process_answer: {e}")
        import traceback
        traceback.print_exc()
        return {
            "answer": f"An error occurred while processing your question: {str(e)}",
            "error": str(e),
            "sources": []
        }
        
# ------------------------------------------------------
# LLM QUERY USING GROQ
# ------------------------------------------------------
def generate_llm_answer(context, query):
    """Generate a formatted answer using Groq LLM"""
    try:
        if not context or not context.strip():
            return "I couldn't find any relevant information in the documents to answer your question."
        
        prompt = f"""You are a helpful AI assistant that answers questions based on provided documents.

CONTEXT FROM DOCUMENTS:
{context}

USER QUESTION: {query}

INSTRUCTIONS:
1. Read the context carefully and provide a clear, well-structured answer
2. Use ONLY information from the context provided
3. Format your response with:
   - A brief direct answer first
   - Key points in bullet format if applicable
   - Clear explanations
4. If the context doesn't contain enough information, say so clearly
5. Do NOT make up or hallucinate any information
6. Be concise but comprehensive

Please provide a well-formatted answer:"""

        # Call Groq API
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that provides clear, structured answers based on provided documents. Always format your responses for easy reading."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=1000,
            top_p=0.9
        )

        answer = response.choices[0].message.content
        
        # Ensure we got a valid response
        if not answer or len(answer.strip()) < 10:
            return format_fallback_answer(context, query)
        
        return answer
    
    except Exception as e:
        print(f"❌ Error generating LLM answer: {e}")
        print(f"Error type: {type(e).__name__}")
        return format_fallback_answer(context, query)


def format_fallback_answer(context, query):
    """Format a fallback answer when LLM fails"""
    summary = context[:500] + "..." if len(context) > 500 else context
    
    return f"""**Answer to: {query}**

Based on the retrieved documents:

{summary}

*Note: This is a direct extract from your documents. For a more refined answer, please check the system logs.*"""