import openai
import numpy as np
import os
from dotenv import load_dotenv
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import json
import time

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to generate embeddings
def generate_embeddings(text, model="text-embedding-3-large"):
    if not text.strip():
        return [0] * 1536  # Return a zero vector for empty or invalid text
    response = openai.Embedding.create(input=text, model=model)
    return response['data'][0]['embedding']

def generate_embeddings_batch(texts, model="text-embedding-3-large", batch_size=50):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        while True:
            try:
                response = openai.Embedding.create(input=batch, model=model)
                embeddings.extend([item['embedding'] for item in response['data']])
                break
            except openai.error.RateLimitError as e:
                print(f"Rate limit reached: {e}. Retrying after delay...")
                time.sleep(5)  # Wait 5 seconds before retrying
    return embeddings

# Function to compute cosine similarity
def compute_similarity(resume_embedding, job_embedding):
    resume_vector = np.array(resume_embedding).reshape(1, -1)
    job_matrix = np.array(job_embedding)
    similarities = cosine_similarity(resume_vector, job_matrix)
    return similarities.flatten()

# Load job listings CSV
def load_job_listings(csv_path):
    if not os.path.exists(csv_path) or os.stat(csv_path).st_size == 0:
        raise FileNotFoundError(f"Job listings file '{csv_path}' does not exist or is empty.")
    return pd.read_csv(csv_path)

# Rank jobs based on similarity
def rank_jobs(resume_embedding, jobs_df, text_column="description"):
    job_texts = jobs_df[text_column].fillna(" ").tolist()
    job_embeddings = generate_embeddings_batch(job_texts)
    similarities = cosine_similarity([resume_embedding], job_embeddings)[0]
    jobs_df['similarity'] = similarities
    ranked_jobs = jobs_df.sort_values(by='similarity', ascending=False)
    return ranked_jobs

def main(resume_json_path, jobs_csv_path):
    # Step 1: Extract text from resume
    if not os.path.exists(resume_json_path):
        raise FileNotFoundError(f"Resume file '{resume_json_path}' not found.")
    with open(resume_json_path, "r", encoding="utf-8") as file:
        try:
            resume_data = json.load(file)
            if isinstance(resume_data, str):  # If it's plain text
                resume_text = resume_data
            else:
                raise ValueError("Unexpected JSON format: expected plain text.")
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON file format.")

    if not resume_text.strip():
        raise ValueError("Parsed resume text is empty or invalid.")

    # Step 2: Generate embedding for the resume
    print("Generating embedding for resume...")
    resume_embedding = generate_embeddings(resume_text)
    
    # Step 3: Load job listings
    print("Loading job listings...")
    jobs_df = load_job_listings(jobs_csv_path)
    
    # Step 4: Rank job listings
    print("Ranking jobs...")
    ranked_jobs = rank_jobs(resume_embedding, jobs_df, text_column="description")

    return ranked_jobs

if __name__ == "__main__":
    resume_json_path = "parsed_resume.json"
    jobs_csv_path = "job_listings.csv"
    try:
        main(resume_json_path, jobs_csv_path)
    except Exception as e:
        print(f"Error: {e}")