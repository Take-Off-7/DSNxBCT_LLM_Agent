# DSNxBCT_LLM_Agent

LLM-powered User Modeling & Review Generation and Recommendation System for the DSN × BCT Data & AI Hackathon 3.0.

---

# 🧠 Overview

This project builds a **user-aware LLM agent system** that learns from Yelp-style data and generates:

- ⭐ Personalized rating predictions  
- ✍🏾 Realistic review text generation  
- 👤 User behavior modeling  
- 🎯 Personalized recommendations
- 🌐 FastAPI-based inference service  

It uses **local LLM inference via Ollama** to ensure offline, low-cost deployment.

---

# 🚀 Key Features

## 👤 User Modeling
Learns user behavior from historical reviews:

- Average rating behavior
- Favorite business categories
- Reviewer style classification:
  - strict reviewer
  - balanced reviewer
  - lenient reviewer

---

## ⭐ Review Generation (Task A)

Given:
- `user_id`
- `business_id`

The system:
- Predicts rating (1–5)
- Generates realistic review text
- Adapts tone based on user personality

---

## 🎯 Recommendation System (Task B)

Given:
- `query`
- `user_id`

The system:
- Recommends relevant businesses
- Personalizes results based on user preferences
- Ranks recommendations by relevance

---

## 🌐 API Service (FastAPI)

Fully container-ready REST API for evaluation and demo purposes.

---

# 🧱 System Architecture

Yelp Dataset (raw JSON)
↓
Data Processing Layer
(process_data.py)
↓
Processed CSV Data
↓
User Profiling Engine
(user_profile.py)
↓
LLM Review Generator (Ollama)
(review_generator.py)
↓
FastAPI Service Layer
(app.py)

---

# 📊 Dataset

Yelp Academic Dataset:

- review.json
- business.json
- user.json

Processed into:

data/processed/
├── reviews.csv
├── businesses.csv
└── users.csv


---

## ⚙️ Tech Stack

- Python 3.10+
- FastAPI
- Pandas
- Requests
- Ollama (local LLM runtime)
- llama3.2 model

---

## ⚙️ Full Setup Guide (Manual Deployment)

### Step 1: Clone Repository
```bash
git clone https://github.com/takeoff7/llm-agent.git
cd llm-agent/project
```

### Step 2: Install Ollama
```bash
curl -fsSL https://ollama.com/install.sh | sh
```
### Step 3: Pull Required Model
```bash
ollama pull llama3.2:1b
```

### Step 4: Start Ollama Server
Open a NEW terminal:
```bash
ollama serve
```
Ollama runs by default at:
http://localhost:11434

### Step 5: Run API Docker Container
Open another terminal:
```bash
docker run --network=host \
  -e OLLAMA_URL=http://localhost:11434/api/generate \
  takeoff7/llm-agent:latest
```

## 🔌 API Access & Endpoints

After setup, access the OpenAPI (Swagger UI):

http://localhost:8000/docs

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/` | GET | Health check |
| `/samples` | GET | Example requests |
| `/profile` | POST | Build or fetch user profile |
| `/review` | POST | Generate simulated review for a user and business |
| `/recommend` | POST | Return personalized ranked recommendations |
| `/llm-status` | GET | Check Ollama connection status |

## 🧪 Testing & Usage Guide

⚡ Performance Optimizations

Recent improvements include:

✅ Switched from phi3 → llama3.2:1b / mistral
✅ Reduced prompt size for faster inference
✅ Added request timeout controls
✅ JSON parsing + fallback handling
✅ Optional caching support (recommended for scale)
✅ Lightweight persona-based prompting
🧪 API Endpoints
🔹 Health Check
GET /
🔹 Task A — Review Generation (ID-based)
POST /review
Input:
{
  "user_id": "string",
  "business_id": "string"
}
🔹 Task A — Review Generation (Name-based)
POST /review-by-name
Input:
{
  "user_name": "user_0",
  "business_name": "Starbucks Coffee"
}
🔹 User Profile
POST /profile
🔹 Sample Data (IMPORTANT FOR JUDGES)
GET /samples

Returns:

valid user IDs
sample businesses
🔹 Demo Input Generator
GET /demo-input

Returns ready-to-use test cases.

📁 Project Structure
project/
├── app.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── process_data.py
│
├── models/
│   ├── user_profile.py
│   └── review_generator.py
│
├── retrieval/
├── notebooks/
├── prompts/
├── requirements.txt
└── README.md
🧪 How to Run
1. Install dependencies
pip install -r requirements.txt
2. Process dataset
python data/process_data.py
3. Start Ollama
ollama serve
4. Run API server
uvicorn app:app --reload
5. Open docs
http://127.0.0.1:8000/docs
🧪 How to Test
Step 1 — Get valid inputs
GET /samples

or

GET /demo-input
Step 2 — Run review generation
Example
POST /review
{
  "user_id": "mh_-eMZ6K5RLWhZyISBhwA",
  "business_id": "abc123"
}
🏆 Hackathon Highlights
Real-world Yelp dataset
LLM-powered personalization
Behavioral user modeling
FastAPI production-ready design
Local inference (Ollama) for offline deployment
Clean modular architecture
🚀 Future Improvements
Vector database (FAISS) for retrieval-augmented reasoning
Async LLM inference (non-blocking API)
Caching layer for repeated queries
Multi-agent recommendation system (Task B)
Dockerized deployment for judges
👨🏾‍💻 Author

Built for DSN × BCT Data & AI Hackathon 3.0
Focus: LLM Agents, User Modeling, Recommendation Systems