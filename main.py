# Updated imports
from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
import os
import faiss
import numpy as np
import pickle
import redis
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, String, Integer, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import hashlib
import json

# Load environment variables
load_dotenv()

# Initialize Redis
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=os.getenv("REDIS_PORT", 6379),
    password=os.getenv("REDIS_PASSWORD"),
    decode_responses=True
)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, lambda _, __: HTTPException(status_code=429, detail="Too many requests"))

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# PostgreSQL (NeonDB) Database Connection
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://neondb_owner:npg_YxEL9V0SrQBR@ep-purple-cloud-a56awla7-pooler.us-east-2.aws.neon.tech/neondb?sslmode=require"
)

# Database setup
engine = create_engine(DATABASE_URL, pool_size=20, max_overflow=10)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

# OpenAI client
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Cache models
class FaissCache:
    @staticmethod
    def get_key(query: str) -> str:
        return f"faiss:{hashlib.md5(query.encode()).hexdigest()}"

    def get(self, query: str):
        cached = redis_client.get(self.get_key(query))
        return json.loads(cached) if cached else None

    def set(self, query: str, result: str):
        redis_client.setex(self.get_key(query), 3600, json.dumps(result))  # 1 hour TTL

class ChatCache:
    @staticmethod
    def get_key(character: str, message: str) -> str:
        return f"chat:{character}:{hashlib.md5(message.encode()).hexdigest()}"

    def get(self, character: str, message: str):
        return redis_client.get(self.get_key(character, message))

    def set(self, character: str, message: str, response: str):
        redis_client.setex(self.get_key(character, message), 21600, response)  # 6 hours TTL
        self._store_history(character, message, response)

    def _store_history(self, character: str, message: str, response: str):
        history_key = f"history:{character}"
        redis_client.lpush(history_key, json.dumps({"message": message, "response": response}))
        redis_client.ltrim(history_key, 0, 4)  # Keep last 5 items

faiss_cache = FaissCache()
chat_cache = ChatCache()

# Database model
class MovieScript(Base):
    __tablename__ = "movie_scripts"
    id = Column(Integer, primary_key=True, index=True)
    character = Column(String, index=True)
    dialogue = Column(String)

# Create database tables
Base.metadata.create_all(bind=engine)

# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# FAISS initialization (same as before)
try:
    index = faiss.read_index("faiss_index.bin")
    with open("database.pkl", "rb") as f:
        database = pickle.load(f)
except:
    index = faiss.IndexFlatIP(1536)
    database = []

# Function to normalize vectors
def normalize(vec):
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm

# Function to convert text into vector embeddings (with normalization)
def get_embedding(text):
    response = openai_client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    vec = np.array(response.data[0].embedding, dtype=np.float32)
    return normalize(vec)

# Function to search for the closest matching dialogue in FAISS
def search_faiss(query):
    if len(database) == 0:
        print("❌ FAISS is empty! No movie scripts found.")
        return None

    query_embedding = get_embedding(query)
    distances, idx = index.search(np.array([query_embedding]), 5)  # Retrieve top 5 matches

    print(f"🔍 FAISS Search Results - Distances: {distances[0]}, Indexes: {idx[0]}")

    best_match = None
    for i, distance in enumerate(distances[0]):
        if distance < 0.40:  # More flexible threshold
            best_match = database[idx[0][i]]
            break

    if best_match:
        print(f"✅ FAISS found a good match: {best_match}")
        return best_match
    else:
        print("⚠️ No exact match found, returning the closest available script dialogue.")
        return database[idx[0][0]]  # Always return the best available match

# Request model for chat endpoint
class ChatRequest(BaseModel):
    character: str
    user_message: str

# Chat API endpoint
@app.post("/chat")
@limiter.limit("5/second")
def chat(request: Request, chat_data: ChatRequest, db: Session = Depends(get_db)):
    # Check cache first
    cached_response = chat_cache.get(chat_data.character, chat_data.user_message)
    if cached_response:
        return {"response": cached_response}

    # Process query
    try:
        script_response = search_faiss(chat_data.user_message)
        messages = [
            {"role": "system", "content": f"You are {chat_data.character}. Previous line: {script_response}"},
            {"role": "user", "content": chat_data.user_message}
        ]

        # OpenAI call
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=messages
        )
        
        result = response.choices[0].message.content
        chat_cache.set(chat_data.character, chat_data.user_message, result)
        return {"response": result}

    except openai.OpenAIError as e:
        raise HTTPException(status_code=503, detail=f"OpenAI error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "redis": redis_client.ping() is True,
        "database": check_db_connection(),
        "faiss": index.ntotal > 0
    }

def check_db_connection():
    try:
        with SessionLocal() as session:
            session.execute(text("SELECT 1"))
            return True
    except Exception as e:
        print(f"Database connection error: {e}")
        return False

# Run server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        limit_concurrency=1000,
        timeout_keep_alive=300
    )