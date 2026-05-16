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

## 📁 Project Structure
```text
project/
├── app.py
├── api/                  # FastAPI routes
├── data/                 # Raw + processed data + embeddings
├── models/               # Rating, ranking, review & behavior models
├── rag/                  # Response generation (RAG pipeline)
├── recommender/          # Recommendation engine
├── retrieval/            # FAISS + embedding search
├── training/             # Model training scripts
├── evaluation/           # Evaluation metrics & scoring
├── prompts/              # LLM prompts
├── utils/                # Helper utilities
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```
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

# 🔄 System Pipeline
```text
Yelp Dataset (Raw JSON)
        ↓
Data Processing Layer (process_data.py)
        ↓
Processed Data (CSV + embeddings)
        ↓
User Profiling Engine (user_profile.py)
        ↓
LLM Review Generator (review_generator.py, Ollama)
        ↓
Recommendation Engine (recommender.py)
        ↓
FastAPI Service Layer (app.py)
```

---

# 📊 Dataset

**Yelp Open Dataset** (real-world business reviews and user interactions):

- review.json
- business.json
- user.json

Processed into:
```text
data/processed/
├── reviews.csv
├── businesses.csv
└── users.csv
```
---

## ⚙️ Tech Stack

- 🐍 Python
- ⚡ FastAPI
- 🤖 Ollama (LLM)
- 🔍 FAISS
- 🧠 Pandas, NumPy
- 📊 Scikit-learn
- 🐳 Docker
- 🗄️ JSON / CSV

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

---

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

---

## 🧪 Testing & Usage Guide

### 🔹 Testing Review Generation (`POST /review`)

- Call `/samples` to get valid test IDs  
- Copy a `user_id` and `business_id`  
- Send request:

```json
{
  "user_id": "FJf1k333aqmmMaMTv-CFNA",
  "business_id": "G6lbDeRY_ZpD7FS5dL3qJw"
}
```

### 🔹 Testing Recommendations (POST /recommend)
- Call `/samples` to get valid `user_id`  
- Provide a query along with the `user_id`
- Send request:

```json
{
  "query": "recommend a coffee shop",
  "user_id": "lcp3WgYyYRfcqewpilwmyg"
}
```

---

## ⚡ Performance Optimizations

### Recent improvements include:

✅ Switched from openAI → ollama
✅ Reduced prompt size for faster inference
✅ Added request timeout controls
✅ JSON parsing + fallback handling
✅ Optional caching support (recommended for scale)
✅ Lightweight persona-based prompting




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