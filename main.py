from fastapi import FastAPI, Depends
from pydantic import BaseModel
import openai
import os
import faiss
import numpy as np
import pickle
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, String, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# Load environment variables
load_dotenv()

# PostgreSQL (NeonDB) Database Connection
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://neondb_owner:npg_YxEL9V0SrQBR@ep-purple-cloud-a56awla7-pooler.us-east-2.aws.neon.tech/neondb?sslmode=require"
)

# Database setup
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

# OpenAI API setup
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key is missing. Add it to .env file.")

openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

# FastAPI app instance
app = FastAPI()

# MovieScript database model
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

# Try to load FAISS index from file (to persist across restarts)
try:
    index = faiss.read_index("faiss_index.bin")
    with open("database.pkl", "rb") as f:
        database = pickle.load(f)
    print(f"✅ Loaded FAISS index with {index.ntotal} dialogues.")
except Exception as e:
    print(f"⚠️ Could not load FAISS index: {e}")
    # Use cosine similarity for better text matching
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

# Function to fetch movie script dialogue from NeonDB
def get_script_dialogue(db: Session, character: str):
    script = db.query(MovieScript).filter(MovieScript.character == character).first()
    return script.dialogue if script else None

# Function to search for the closest matching dialogue in FAISS using cosine similarity
def search_faiss(query):
    if len(database) == 0:
        print("❌ FAISS is empty! No movie scripts found.")
        return None

    query_embedding = get_embedding(query)
    distances, idx = index.search(np.array([query_embedding]), 5)  # Retrieve top 5 matches

    print(f"🔍 FAISS Search Results - Distances: {distances[0]}, Indexes: {idx[0]}")

    # Adjusted threshold for better response accuracy
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
async def chat(request: ChatRequest, db: Session = Depends(get_db)):
    try:
        script_response = search_faiss(request.user_message)

        if script_response:
            messages = [
                {"role": "system", "content": f"You are {request.character}. Here’s a movie dialogue you’ve said before: \"{script_response}\""},
                {"role": "user", "content": request.user_message}
            ]
        else:
            messages = [
                {"role": "system", "content": f"You are {request.character}."},
                {"role": "user", "content": request.user_message}
            ]

        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=messages
        )
        return {"response": response.choices[0].message.content}

    except openai.OpenAIError as e:
        return {"error": f"OpenAI API error: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}



# Run server only if script is executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
