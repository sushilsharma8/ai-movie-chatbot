from fastapi import FastAPI, Depends
from pydantic import BaseModel
import openai
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, String, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# Load environment variables
load_dotenv()

# Use NeonDB PostgreSQL connection
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://neondb_owner:npg_YxEL9V0SrQBR@ep-purple-cloud-a56awla7-pooler.us-east-2.aws.neon.tech/neondb?sslmode=require")

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

# Request model for chat endpoint
class ChatRequest(BaseModel):
    character: str
    user_message: str

# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Function to fetch movie script dialogue from NeonDB
def get_script_dialogue(db: Session, character: str):
    script = db.query(MovieScript).filter(MovieScript.character == character).first()
    return script.dialogue if script else None

# Chat API endpoint
@app.post("/chat")
async def chat(request: ChatRequest, db: Session = Depends(get_db)):
    try:
        # Check if a real movie dialogue exists in NeonDB
        script_response = get_script_dialogue(db, request.character)

        if script_response:
            return {"response": script_response}

        # If no movie script is found, generate response using OpenAI
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": f"You are {request.character}."},
                {"role": "user", "content": request.user_message}
            ]
        )
        return {"response": response.choices[0].message.content}

    except Exception as e:
        return {"error": str(e)}

# Run server only if script is executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
