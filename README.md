# AI Movie Chatbot - Backend

## 📌 Overview

This is the backend service for the AI Movie Chatbot, which allows users to chat with AI-powered versions of their favorite movie characters. It supports real-time responses using OpenAI's GPT-4 and Google's Gemini models, optimized with FAISS for efficient movie dialogue retrieval.

---

## 🚀 Features

- **FastAPI-based backend** for high-performance API handling.
- **Dual AI Model Support:** OpenAI GPT-4 and Google Gemini.
- **FAISS Vector Search:** Retrieves relevant movie dialogues efficiently.
- **Redis Caching:** Speeds up responses and reduces API costs.
- **PostgreSQL Database:** Stores chat history for each user.
- **Rate Limiting:** Protects the API from excessive requests.
- **CORS Enabled:** Allows frontend communication.
- **Prometheus & Grafana Monitoring:** Tracks API performance in production.

---

## 🛠️ Tech Stack

- **Backend Framework:** FastAPI (Python)
- **Database:** PostgreSQL (Hosted on NeonDB)
- **Vector Search:** FAISS
- **Caching:** Redis
- **AI Models:** OpenAI GPT-4, Google Gemini
- **Monitoring:** Prometheus + Grafana
- **Hosting:** Railway

---

## 🔧 Setup & Installation

### 1️⃣ **Clone the Repository**

```sh
git clone https://github.com/sushilsharma8/ai-movie-chatbot-backend.git
cd ai-movie-chatbot-backend
```

### 2️⃣ **Set Up Virtual Environment** (Recommended)

```sh
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3️⃣ **Install Dependencies**

```sh
pip install -r requirements.txt
```

### 4️⃣ **Set Up Environment Variables**

Create a `.env` file in the project root and add:

```ini
DATABASE_URL=your_postgresql_connection_string
OPENAI_API_KEY=your_openai_api_key
GEMINI_API_KEY=your_google_gemini_api_key
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=your_redis_password
```

### 5️⃣ **Scrape Movie Scripts**
Before running the chatbot, scrape and store movie scripts using:

```sh
python scrape_imsdb.py
```
This will fetch movie scripts and store them in the PostgreSQL database.

### 6️⃣ **Populate FAISS Vector Index**
Once movie scripts are stored, generate embeddings and populate FAISS:

```sh
python populate_db.py
```
This step converts dialogues into vector embeddings and stores them in FAISS for faster retrieval.

### 7️⃣ **Start Redis Server**
Ensure Redis is running locally or use a cloud-based Redis service:

```sh
redis-server
```

### 8️⃣ **Run the Backend Locally**

```sh
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
The API will be accessible at: **`http://localhost:8000`**


## 📖 API Documentation

The backend provides the following API endpoints:

### 🔹 **1. Chat with AI**

**Endpoint:** `POST /chat`
**Description:** Allows users to chat with AI-powered movie characters.
**Request Body:**

```json
{
  "user_id": "user123",
  "character": "Iron Man",
  "user_message": "Hello!"
}
```

**Response:**

```json
{
  "response": "Hello! I am Iron Man."
}
```

### 🔹 **2. Fetch Chat History**

**Endpoint:** `GET /chat-history/{user_id}`
**Description:** Retrieves the last 10 chat messages of a user.
**Response:**

```json
[
  {
    "character": "Iron Man",
    "user_message": "Hello!",
    "bot_response": "Hello! I am Iron Man. How can I assist you today?",
    "timestamp": "2024-03-18T12:34:56"
  }
]
```

### 🔹 **3. Health Check**

**Endpoint:** `GET /health`
**Description:** Verifies that all backend services are running properly.
**Response:**

```json
{
  "status": "ok",
  "redis": true,
  "database": true,
  "faiss": true
}
```
### 🔹 **4. Postman Collection**
[AI Movie Character Chatbot - Avengers- Endgame Test Cases.postman_collection.json](https://github.com/user-attachments/files/18815931/AI.Movie.Character.Chatbot.-.Avengers-.Endgame.Test.Cases.postman_collection.json)


## AI Movie Chatbot

![AI Movie Chatbot UI](https://raw.githubusercontent.com/sushilsharma8/ai-movie-chatbot/main/Screenshot1.png)

This is a screenshot of the AI Movie Chatbot in action.


## 🛠️ Troubleshooting

### **Common Issues & Fixes**

❌ *CORS error?* → Ensure FastAPI CORS settings allow your frontend’s origin.\
❌ *Database not connecting?* → Verify the `DATABASE_URL` in the `.env` file.\
❌ *Redis not working?* → Check if Redis is running locally (`redis-server`).

For further issues, create a GitHub issue in the repository.

---

