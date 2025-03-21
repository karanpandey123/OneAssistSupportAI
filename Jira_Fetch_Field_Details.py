from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import requests
import openai
import json
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load API keys from an external file (e.g., config.json)
with open("config.json") as f:
    config = json.load(f)

JIRA_API_URL = config["JIRA_SERVER"]
JIRA_USERNAME = config["JIRA_USER"]
JIRA_API_KEY = config["JIRA_API_TOKEN"]
OPENAI_API_KEY = config["OPENAI_API_KEY"]

# Initialize OpenAI API
openai.api_key = OPENAI_API_KEY

# FastAPI app initialization
app = FastAPI()

# Define request body structure
class TicketRequest(BaseModel):
    summary: str
    description: str

# Fetch tickets from Jira
def fetch_jira_tickets():
    headers = {
        "Authorization": f"Basic {JIRA_API_KEY}",
        "Content-Type": "application/json"
    }
    response = requests.get(JIRA_API_URL, headers=headers)

    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Failed to fetch Jira tickets")

    tickets = response.json()["issues"]
    df = pd.DataFrame([
        {"Key": ticket["key"], "Summary": ticket["fields"]["summary"], "Created": ticket["fields"]["created"]}
        for ticket in tickets
    ])
    
    df["Created"] = pd.to_datetime(df["Created"], errors='coerce')
    return df

# Predefined high-priority ticket patterns
high_priority_keywords = ["oasys not working", "unite not working", "crm not working"]

# Function to check for duplicate tickets
def check_duplicate(ticket_summary, created_time, df):
    duplicate_keys = []
    
    for keyword in high_priority_keywords:
        if keyword in ticket_summary.lower():
            one_hour_ago = created_time - timedelta(hours=1)
            duplicates = df[(df["Created"] >= one_hour_ago) & 
                            (df["Created"] <= created_time) & 
                            (df["Summary"].str.contains(keyword, case=False, na=False))]
            if not duplicates.empty:
                duplicate_keys = duplicates["Key"].tolist()
                return "Duplicate", duplicate_keys

    # Step 2: Cosine Similarity on same-day tickets
    same_day_tickets = df[df["Created"].dt.date == created_time.date()]
    if not same_day_tickets.empty:
        vectorizer = TfidfVectorizer()
        text_corpus = same_day_tickets["Summary"].fillna("").tolist() + [ticket_summary]
        tfidf_matrix = vectorizer.fit_transform(text_corpus)
        similarity_matrix = cosine_similarity(tfidf_matrix)

        # Compare last row (current ticket) with previous tickets
        similarity_scores = similarity_matrix[-1, :-1]
        if max(similarity_scores, default=0) > 0.8:  # Similarity threshold
            duplicate_idx = similarity_scores.argmax()
            duplicate_keys = same_day_tickets.iloc[[duplicate_idx]]["Key"].tolist()
            return "Duplicate", duplicate_keys
    
    return "Not Duplicate", []

# Function to classify the ticket using OpenAI GPT
def classify_ticket_with_gpt(summary, description):
    prompt = f"""
    Categorize the following IT support ticket:
    Summary: {summary}
    Description: {description}
    
    Categories:
    1. Rejected/No User Response
    2. Data Processing
    3. User Clarification
    4. Maintenance
    5. System Issue
    6. Change Request
    7. Tech Support Maintenance
    
    Return only the category name.
    """

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are an IT support assistant."},
                  {"role": "user", "content": prompt}]
    )

    category = response["choices"][0]["message"]["content"].strip()
    return category

# API Endpoint to process a new ticket
@app.post("/process_ticket")
async def process_ticket(ticket: TicketRequest):
    df = fetch_jira_tickets()
    created_time = datetime.utcnow()

    # Check for duplicate tickets
    duplicate_status, duplicate_keys = check_duplicate(ticket.summary, created_time, df)

    # If not a duplicate, classify the ticket
    category = "N/A"
    if duplicate_status == "Not Duplicate":
        category = classify_ticket_with_gpt(ticket.summary, ticket.description)

    return {
        "duplicate_status": duplicate_status,
        "duplicate_keys": duplicate_keys,
        "category": category
    }

