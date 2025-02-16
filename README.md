[AI Movie Character Chatbot - Avengers- Endgame Test Cases.postman_collection.json](https://github.com/user-attachments/files/18815931/AI.Movie.Character.Chatbot.-.Avengers-.Endgame.Test.Cases.postman_collection.json)# AI Movie Chatbot - Backend

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
- **Hosting:** Vercel

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

### 5️⃣ **Start Redis Server**

Ensure Redis is running locally or use a managed Redis service.

```sh
redis-server
```

### 6️⃣ **Run the Backend Locally**

```sh
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be accessible at: **`http://localhost:8000`**

---

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

---[Uploading AI {
	"info": {
		"_postman_id": "2aacfcae-c299-4729-9f00-fc1715091a2f",
		"name": "AI Movie Character Chatbot - Avengers: Endgame Test Cases",
		"schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json",
		"_exporter_id": "37069919"
	},
	"item": [
		{
			"name": "Iron Man - Do you think we can really und...",
			"request": {
				"method": "POST",
				"header": [
					{
						"key": "Content-Type",
						"value": "application/json"
					}
				],
				"body": {
					"mode": "raw",
					"raw": "{\n    \"user_id\": \"user001\",\n    \"character\": \"Iron Man\",\n    \"user_message\": \"Do you think we can really undo the Snap?\"\n}"
				},
				"url": {
					"raw": "https://ai-movie-chatbot-production.up.railway.app/chat",
					"protocol": "https",
					"host": [
						"ai-movie-chatbot-production",
						"up",
						"railway",
						"app"
					],
					"path": [
						"chat"
					]
				}
			},
			"response": []
		},
		{
			"name": "Captain America - How do we bring everyone back?...",
			"request": {
				"method": "POST",
				"header": [
					{
						"key": "Content-Type",
						"value": "application/json"
					}
				],
				"body": {
					"mode": "raw",
					"raw": "{\n    \"user_id\": \"user002\",\n    \"character\": \"Captain America\",\n    \"user_message\": \"How do we bring everyone back?\"\n}"
				},
				"url": {
					"raw": "https://ai-movie-chatbot-production.up.railway.app/chat",
					"protocol": "https",
					"host": [
						"ai-movie-chatbot-production",
						"up",
						"railway",
						"app"
					],
					"path": [
						"chat"
					]
				}
			},
			"response": []
		},
		{
			"name": "Thanos - Do you regret what you did?...",
			"request": {
				"method": "POST",
				"header": [
					{
						"key": "Content-Type",
						"value": "application/json"
					}
				],
				"body": {
					"mode": "raw",
					"raw": "{\n    \"user_id\": \"user003\",\n    \"character\": \"Thanos\",\n    \"user_message\": \"Do you regret what you did?\"\n}"
				},
				"url": {
					"raw": "https://ai-movie-chatbot-production.up.railway.app/chat",
					"protocol": "https",
					"host": [
						"ai-movie-chatbot-production",
						"up",
						"railway",
						"app"
					],
					"path": [
						"chat"
					]
				}
			},
			"response": []
		},
		{
			"name": "Thor - I failed to stop Thanos. How d...",
			"request": {
				"method": "POST",
				"header": [
					{
						"key": "Content-Type",
						"value": "application/json"
					}
				],
				"body": {
					"mode": "raw",
					"raw": "{\n    \"user_id\": \"user004\",\n    \"character\": \"Thor\",\n    \"user_message\": \"I failed to stop Thanos. How do I fix this?\"\n}"
				},
				"url": {
					"raw": "https://ai-movie-chatbot-production.up.railway.app/chat",
					"protocol": "https",
					"host": [
						"ai-movie-chatbot-production",
						"up",
						"railway",
						"app"
					],
					"path": [
						"chat"
					]
				}
			},
			"response": []
		},
		{
			"name": "Hulk - How did you merge Banner and H...",
			"request": {
				"method": "POST",
				"header": [
					{
						"key": "Content-Type",
						"value": "application/json"
					}
				],
				"body": {
					"mode": "raw",
					"raw": "{\n    \"user_id\": \"user005\",\n    \"character\": \"Hulk\",\n    \"user_message\": \"How did you merge Banner and Hulk?\"\n}"
				},
				"url": {
					"raw": "https://ai-movie-chatbot-production.up.railway.app/chat",
					"protocol": "https",
					"host": [
						"ai-movie-chatbot-production",
						"up",
						"railway",
						"app"
					],
					"path": [
						"chat"
					]
				}
			},
			"response": []
		},
		{
			"name": "Black Widow - We’ve lost so much. How do we ...",
			"request": {
				"method": "POST",
				"header": [
					{
						"key": "Content-Type",
						"value": "application/json"
					}
				],
				"body": {
					"mode": "raw",
					"raw": "{\n    \"user_id\": \"user006\",\n    \"character\": \"Black Widow\",\n    \"user_message\": \"We\\u2019ve lost so much. How do we move forward?\"\n}"
				},
				"url": {
					"raw": "https://ai-movie-chatbot-production.up.railway.app/chat",
					"protocol": "https",
					"host": [
						"ai-movie-chatbot-production",
						"up",
						"railway",
						"app"
					],
					"path": [
						"chat"
					]
				}
			},
			"response": []
		},
		{
			"name": "Hawkeye - Do you think Nat would be prou...",
			"request": {
				"method": "POST",
				"header": [
					{
						"key": "Content-Type",
						"value": "application/json"
					}
				],
				"body": {
					"mode": "raw",
					"raw": "{\n    \"user_id\": \"user007\",\n    \"character\": \"Hawkeye\",\n    \"user_message\": \"Do you think Nat would be proud of us?\"\n}"
				},
				"url": {
					"raw": "https://ai-movie-chatbot-production.up.railway.app/chat",
					"protocol": "https",
					"host": [
						"ai-movie-chatbot-production",
						"up",
						"railway",
						"app"
					],
					"path": [
						"chat"
					]
				}
			},
			"response": []
		},
		{
			"name": "Ant-Man - Time travel? Are you serious?...",
			"request": {
				"method": "POST",
				"header": [
					{
						"key": "Content-Type",
						"value": "application/json"
					}
				],
				"body": {
					"mode": "raw",
					"raw": "{\n    \"user_id\": \"user008\",\n    \"character\": \"Ant-Man\",\n    \"user_message\": \"Time travel? Are you serious?\"\n}"
				},
				"url": {
					"raw": "https://ai-movie-chatbot-production.up.railway.app/chat",
					"protocol": "https",
					"host": [
						"ai-movie-chatbot-production",
						"up",
						"railway",
						"app"
					],
					"path": [
						"chat"
					]
				}
			},
			"response": []
		},
		{
			"name": "Doctor Strange - Is there really only one way t...",
			"request": {
				"method": "POST",
				"header": [
					{
						"key": "Content-Type",
						"value": "application/json"
					}
				],
				"body": {
					"mode": "raw",
					"raw": "{\n    \"user_id\": \"user009\",\n    \"character\": \"Doctor Strange\",\n    \"user_message\": \"Is there really only one way this ends?\"\n}"
				},
				"url": {
					"raw": "https://ai-movie-chatbot-production.up.railway.app/chat",
					"protocol": "https",
					"host": [
						"ai-movie-chatbot-production",
						"up",
						"railway",
						"app"
					],
					"path": [
						"chat"
					]
				}
			},
			"response": []
		},
		{
			"name": "Doctor Strange",
			"request": {
				"method": "POST",
				"header": [
					{
						"key": "Content-Type",
						"value": "application/json"
					}
				],
				"body": {
					"mode": "raw",
					"raw": "{\n    \"user_id\": \"user009\",\n    \"character\": \"Doctor Strange\",\n    \"user_message\": \"Is there really only one way this ends?\"\n}"
				},
				"url": {
					"raw": "https://ai-movie-chatbot-production.up.railway.app/chat",
					"protocol": "https",
					"host": [
						"ai-movie-chatbot-production",
						"up",
						"railway",
						"app"
					],
					"path": [
						"chat"
					]
				}
			},
			"response": []
		},
		{
			"name": "Health Check",
			"request": {
				"method": "GET",
				"header": [],
				"url": {
					"raw": "http://localhost:8000/health",
					"protocol": "http",
					"host": [
						"localhost"
					],
					"port": "8000",
					"path": [
						"health"
					]
				},
				"description": "Generated from cURL: curl http://localhost:8000/health"
			},
			"response": []
		}
	]
}Movie Character Chatbot - Avengers- Endgame Test Cases.postman_collection.json…]()


## 🛠️ Troubleshooting

### **Common Issues & Fixes**

❌ *CORS error?* → Ensure FastAPI CORS settings allow your frontend’s origin.\
❌ *Database not connecting?* → Verify the `DATABASE_URL` in the `.env` file.\
❌ *Redis not working?* → Check if Redis is running locally (`redis-server`).

For further issues, create a GitHub issue in the repository.

---

