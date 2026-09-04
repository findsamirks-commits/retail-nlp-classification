# 🛒 Retail NLP Triage API

An automated Natural Language Processing pipeline that reads unstructured customer grocery reviews, scores emotional sentiment, and routes tickets to the correct retail department.

## Features
* **Sentiment Scoring:** Uses NLTK's VADER to assign a mathematical emotion score (-1.0 to +1.0) to raw feedback.
* **Automated Action Flags:** Triggers immediate business rules based on sentiment severity (e.g., 🚨 Escalate to Support, ⚠️ Operations Alert).
* **Department Classification:** Uses a supervised TF-IDF and Naive Bayes machine learning pipeline to categorize tickets into specific business units (Fruits & Vegetables, Dairy, Bakery, Tech & Logistics).

## Tech Stack
* **Backend:** FastAPI, Uvicorn, Pydantic
* **Machine Learning:** Scikit-Learn, Joblib
* **NLP:** NLTK (VADER Lexicon)

## Local Setup & Testing
1. Install the required dependencies:
   ```bash
   pip install fastapi uvicorn scikit-learn joblib pydantic nltk
   ```
2. Start the local development server:
```
python -m uvicorn app:app --reload
```
3. Navigate to http://127.0.0.1:8000/docs in your browser to interactively test the /triage endpoint via the Swagger UI.

