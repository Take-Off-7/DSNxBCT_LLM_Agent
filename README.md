# DSNxBCT_LLM_Agent

LLM-powered User Modeling & Review Generation System for the DSN × BCT Data & AI Hackathon 3.0.

---

# 🧠 Overview

This project builds a **user-aware LLM agent system** that learns from Yelp-style data and generates:

- ⭐ Personalized rating predictions  
- ✍🏾 Realistic review text generation  
- 👤 User behavior modeling  
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

# ⚙️ Tech Stack

- Python 3.10+
- FastAPI
- Pandas
- Requests
- Ollama (local LLM runtime)
- llama3.2 / mistral models

---

# 🧠 LLM Setup

This project uses **Ollama for local inference**:

### Install model
```bash
ollama pull llama3.2:1b

or faster alternative:

ollama pull mistral

Start Ollama server

ollama serve

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