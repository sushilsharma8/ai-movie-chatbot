from fastapi import FastAPI
from pydantic import BaseModel
import openai
import os
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Ensure the key is set
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key is missing. Add it to .env file.")

openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)  # Use OpenAI class properly

app = FastAPI()

class ChatRequest(BaseModel):
    character: str
    user_message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": f"You are {request.character}."},
                {"role": "user", "content": request.user_message}
            ]
        )
        return {"response": response.choices[0].message.content}  # Corrected response parsing
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
