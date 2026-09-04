from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download VADER lexicon for local sentiment scoring
nltk.download('vader_lexicon', quiet=True)

# Load the trained machine learning pipeline
classifier = joblib.load("department_classifier.joblib")
sentiment_analyzer = SentimentIntensityAnalyzer()

app = FastAPI(title="Retail NLP Triage API")

class TicketInput(BaseModel):
    review_text: str

@app.post("/triage")
def triage_ticket(ticket: TicketInput):
    # 1. Predict the target department
    dept = classifier.predict([ticket.review_text])[0]
    
    # 2. Score the emotional tone
    score = sentiment_analyzer.polarity_scores(ticket.review_text)['compound']
    
    # 3. Assign automated business actions
    if score <= -0.5:
        action = "🚨 Escalate to Support"
    elif score < 0:
        action = "⚠️ Operations Alert"
    elif score > 0.4:
        action = "🌟 Positive Testimonial"
    else:
        action = "✅ Standard Review"
        
    return {
        "predicted_department": dept,
        "sentiment_score": score,
        "action_flag": action
    }