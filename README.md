# DSNxBCT_LLM_Agent

Download dataset from Kaggle:
https://www.kaggle.com/datasets/adamamer2001/yelp-complete-open-dataset-2024

Place files in:
project/data/raw/



.🧠 LLM Agent Hackathon — Task A (User Modeling & Review Generation)
📌 Overview

This project is part of the DSN × BCT Data & AI Hackathon 3.0.

It builds a user-aware AI agent system that:

models user behavior from Yelp reviews
predicts ratings for unseen businesses
generates contextual review text
exposes everything through a FastAPI service
includes human-friendly testing endpoints for easy evaluation
🚀 What This System Does
👤 User Modeling
Learns behavior from historical reviews
Builds user profiles:
average rating
favorite categories
reviewer style (strict / balanced / lenient)
⭐ Review Generation

Given a user and business, the system:

predicts a rating
generates a natural-language review
adapts tone based on user behavior
🌐 API Layer (FastAPI)

Provides endpoints for:

structured testing (IDs)
human-friendly testing (names)
sample data discovery (for judges)
🧱 System Architecture
Yelp Dataset (raw JSON)
        ↓
Data Processing Layer
(process_data.py)
        ↓
Structured CSV Data
        ↓
User Profiling Engine
(user_profile.py)
        ↓
Review Generation Engine
(review_generator.py)
        ↓
FastAPI Service Layer
(app.py)
📊 Dataset Used

We use the Yelp Academic Dataset:

review.json
business.json
user.json

Processed into:

data/processed/
├── reviews.csv
├── businesses.csv
└── users.csv
⚙️ Core Components
1. Data Processing

File:

data/process_data.py
Purpose:
Loads raw Yelp JSON files
Samples dataset for efficiency
Converts into structured CSV format
2. User Profiling Engine

File:

models/user_profile.py
Output:
average rating behavior
favorite categories
user style classification:
strict reviewer
balanced reviewer
lenient reviewer
3. Review Generator

File:

models/review_generator.py
Input:
user_id
business_id
Output:
predicted rating
generated review text
4. API Layer (FastAPI)

File:

app.py
🌐 API Endpoints
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
🔹 Sample Data (IMPORTANT FOR TESTING)
GET /samples
Returns:
valid user IDs
sample business names + IDs
🔹 Demo Input Generator
GET /demo-input
Returns:

Ready-to-use test cases:

{
  "user_id": "...",
  "business_name": "..."
}
🧪 How to Run
1. Install dependencies
pip install -r requirements.txt
2. Process dataset
python data/process_data.py
3. Run API server
uvicorn app:app --reload
4. Open API documentation
http://127.0.0.1:8000/docs
🧪 How to Test
Step 1 — Get valid inputs
GET /samples

or

GET /demo-input
Step 2 — Use in API
Example (ID-based):
POST /review
{
  "user_id": "mh_-eMZ6K5RLWhZyISBhwA",
  "business_id": "abc123"
}
Example (Name-based):
POST /review-by-name
{
  "user_name": "user_0",
  "business_name": "Starbucks Coffee"
}
📁 Project Structure
project/
├── app.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── process_data.py
├── models/
│   ├── user_profile.py
│   └── review_generator.py
├── notebooks/
├── prompts/
├── retrieval/
└── requirements.txt