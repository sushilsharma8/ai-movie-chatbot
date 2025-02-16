# Updated imports
from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
import google.generativeai as genai
import os
import faiss
import numpy as np
import pickle
import redis
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, String, Integer, text, DateTime, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import hashlib
import json
from prometheus_fastapi_instrumentator import Instrumentator

# Load environment variables
load_dotenv()

# Determine Redis connection URL (Railway / Upstash / Local)
REDIS_URL = os.getenv("REDIS_URL")  # Use full Redis URL if provided

if REDIS_URL:
    redis_client = redis.from_url(REDIS_URL, decode_responses=True)  # Cloud Redis (Railway / Upstash)
else:
    redis_client = redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        password=os.getenv("REDIS_PASSWORD", None),
        decode_responses=True
    )  # Local Redis (Docker)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, lambda _, __: HTTPException(status_code=429, detail="Too many requests"))

# ✅ Add your Netlify frontend URL here
origins = [
    "https://dulcet-kataifi-2ea36d.netlify.app",  # Your Netlify URL
    "http://localhost:3000",  # If testing locally
]

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],  # ✅ Allow all HTTP methods
    allow_headers=["*"],  # ✅ Allow all headers
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

# API Clients: OpenAI + Gemini
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not OPENAI_API_KEY or not GEMINI_API_KEY:
    raise ValueError("Missing API Keys. Add OpenAI & Gemini keys in .env file.")

openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)

# FAISS Cache
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

faiss_cache = FaissCache()
chat_cache = ChatCache()

# Database Models
class MovieScript(Base):
    __tablename__ = "movie_scripts"
    id = Column(Integer, primary_key=True, index=True)
    character = Column(String, index=True)
    dialogue = Column(String)

class ChatHistory(Base):
    __tablename__ = "chat_history"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)  # User identifier
    character = Column(String)
    user_message = Column(String)
    bot_response = Column(String)
    timestamp = Column(DateTime, default=func.now())

# Create tables
Base.metadata.create_all(bind=engine)

# Dependency for DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# FAISS initialization
try:
    index = faiss.read_index("faiss_index.bin")
    with open("database.pkl", "rb") as f:
        database = pickle.load(f)
except:
    index = faiss.IndexFlatIP(1536)
    database = []

# Normalize vectors
def normalize(vec):
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm

# Convert text to vector embedding
def get_embedding(text):
    try:
        model = genai.GenerativeModel("gemini-pro-embed")
        response = model.embed_content(text)
        return np.array(response["embedding"], dtype=np.float32)
    except Exception:
        return None  # If Gemini fails, return None

# FAISS search
def search_faiss(query):
    if len(database) == 0:
        return None

    query_embedding = get_embedding(query)
    if query_embedding is None:
        return None  # If embedding failed, return None

    distances, idx = index.search(np.array([query_embedding]), 5)
    return database[idx[0][0]] if distances[0][0] < 0.40 else None

def get_gemini_response(character, user_message, script_response):
    try:
        model = genai.GenerativeModel("gemini-pro")
        
        # ✅ Corrected message format
        messages = [
            {"role": "user", "parts": [{"text": f"You are {character}. Previous line: {script_response}"}]},
            {"role": "user", "parts": [{"text": user_message}]}
        ]
        
        response = model.generate_content(messages)

        # ✅ Ensure response is valid before accessing text
        if response and response.candidates and response.candidates[0].content.parts:
            return response.candidates[0].content.parts[0].text  # Extract response safely
        
        print("⚠️ Gemini returned an empty response.")
        return None  # Return None if Gemini fails

    except Exception as e:
        print(f"⚠️ Gemini API failed: {e}")
        return None  # Prevent crashes by returning None

def get_openai_response(character, user_message, script_response):
    try:
        messages = [
            {"role": "system", "content": f"You are {character}. Previous line: {script_response}"},
            {"role": "user", "content": user_message}
        ]
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=messages
        )
        return response.choices[0].message.content
    except openai.OpenAIError as e:
        print(f"⚠️ OpenAI API failed: {e}")
        if "insufficient_quota" in str(e):
            return "I'm currently experiencing high demand. Please try again later."
        return None  # Instead of crashing, return None


# Request model
class ChatRequest(BaseModel):
    user_id: str
    character: str
    user_message: str

Instrumentator().instrument(app).expose(app)

def save_chat_history(db: Session, user_id: str, character: str, user_message: str, bot_response: str):
    try:
        # print(f"📝 Attempting to save chat history for user_id: {user_id}")  # ✅ Debug log

        chat_entry = ChatHistory(
            user_id=user_id,
            character=character,
            user_message=user_message,
            bot_response=bot_response
        )

        db.add(chat_entry)
        db.commit()  # ✅ Ensures transaction is committed
        db.refresh(chat_entry)  # ✅ Refresh to ensure entry is stored
        # print(f"✅ Chat history saved: {chat_entry.id}")

    except Exception as e:
        db.rollback()  # ✅ Prevents broken transactions
        print(f"⚠️ Failed to save chat history: {e}")




# Retrieve chat history
@app.get("/chat-history/{user_id}")
def get_chat_history(user_id: str, db: Session = Depends(get_db)):
    try:
        print(f"🔍 Fetching chat history for user_id: {user_id}")  # ✅ Debug log

        history = (
            db.query(ChatHistory)
            .filter(ChatHistory.user_id == user_id)
            .order_by(ChatHistory.timestamp.desc())
            .limit(10)
            .all()
        )

        if not history:
            print(f"⚠️ No chat history found for user_id: {user_id}")  # ✅ Debug log
            return {"message": "No chat history found."}

        return [
            {
                "character": chat.character,
                "user_message": chat.user_message,
                "bot_response": chat.bot_response,
                "timestamp": chat.timestamp
            }
            for chat in history
        ]

    except Exception as e:
        print(f"⚠️ Error fetching chat history: {e}")
        return {"error": "Failed to fetch chat history."}

# Chat API
@app.post("/chat")
@limiter.limit("5/second")
def chat(request: Request, chat_data: ChatRequest, db: Session = Depends(get_db)):
    cached_response = chat_cache.get(chat_data.character, chat_data.user_message)
    if cached_response:
        return {"response": cached_response}

    script_response = search_faiss(chat_data.user_message) or "No previous dialogue found."

    # ✅ Try Gemini first
    result = get_gemini_response(chat_data.character, chat_data.user_message, script_response)
    
    # ✅ If Gemini fails, try OpenAI
    if not result:
        result = get_openai_response(chat_data.character, chat_data.user_message, script_response)

    # ✅ If both fail, return a fallback message
    if not result:
        result = "I'm currently unable to generate a response. Please try again later."

    # ✅ Log Chat History Saving
    # print(f"📝 Saving chat history for user_id: {chat_data.user_id}")  # ✅ Debug log
    save_chat_history(db, chat_data.user_id, chat_data.character, chat_data.user_message, result)

    # ✅ Cache response to reduce latency
    chat_cache.set(chat_data.character, chat_data.user_message, result)

    return {"response": result}

# Health check
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
        # print(f"Database connection error: {e}")
        return False

# Run server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
