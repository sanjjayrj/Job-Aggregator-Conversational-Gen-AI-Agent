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

EMBEDDINGS_FILE = "job_embeddings.json"

# Function to generate embeddings for resume
def generate_embeddings(text, model="text-embedding-3-large"):
    if not text.strip():
        return [0] * 1536  # Return a zero vector for empty or invalid text
    response = openai.Embedding.create(input=text, model=model)
    return response['data'][0]['embedding']

def save_resume_embedding(resume_text, filepath="resume_embedding.json"):
    """Generate and save resume embedding."""
    embedding = generate_embeddings(resume_text)
    with open(filepath, "w") as file:
        json.dump(embedding, file)
    return embedding

def load_resume_embedding(filepath="resume_embedding.json"):
    """Load stored resume embedding."""
    if os.path.exists(filepath):
        with open(filepath, "r") as file:
            return json.load(file)
    return None

def generate_embeddings_batch(texts, model="text-embedding-3-large", batch_size=100):
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

def save_embeddings(embeddings, filepath=EMBEDDINGS_FILE):
    """Save embeddings with associated job IDs."""
    if os.path.exists(filepath):
        with open(filepath, "r") as file:
            stored_data = json.load(file)
    else:
        stored_data = {}

    # Update stored embeddings with new ones
    stored_data.update(embeddings)

    with open(filepath, "w") as file:
        json.dump(stored_data, file)

def load_embeddings(filepath=EMBEDDINGS_FILE):
    """Load stored embeddings."""
    if os.path.exists(filepath):
        with open(filepath, "r") as file:
            return json.load(file)
    return {}

def hash_job(job):
    """Generate a unique identifier for a job using its URL or title and company."""
    return f"{job['title']}_{job['company']}".lower().replace(" ", "_")

def embed_new_jobs(jobs_df, stored_embeddings, text_column="description"):
    """Embed only new jobs that are not already in stored_embeddings."""
    new_embeddings = {}
    new_texts = []
    new_ids = []

    for idx, row in jobs_df.iterrows():
        job_id = hash_job(row)
        if job_id not in stored_embeddings:
            # New job found
            new_texts.append(row[text_column])
            new_ids.append(job_id)

    if new_texts:
        print(f"Generating embeddings for {len(new_texts)} new jobs...")
        batch_embeddings = generate_embeddings_batch(new_texts)
        for job_id, embedding in zip(new_ids, batch_embeddings):
            new_embeddings[job_id] = {"embedding": embedding, "data": row.to_dict()}

    return new_embeddings

# Rank jobs based on similarity
def rank_jobs(resume_embedding, jobs_embeddings, text_column="description"):
    similarities = []
    for _, job_data in jobs_embeddings.items():
        job_embedding = np.array(job_data["embedding"]).reshape(1, -1)
        similarity = cosine_similarity([resume_embedding], job_embedding)[0][0]
        similarities.append((job_data["data"], similarity))

    # Convert to DataFrame and sort
    ranked_jobs = pd.DataFrame([data for data, sim in similarities])
    ranked_jobs['similarity'] = [sim for _, sim in similarities]
    return ranked_jobs.sort_values(by="similarity", ascending=False)

# Load job listings CSV
def load_job_listings(csv_path):
    if not os.path.exists(csv_path) or os.stat(csv_path).st_size == 0:
        raise FileNotFoundError(f"Job listings file '{csv_path}' does not exist or is empty.")
    return pd.read_csv(csv_path)

def main(resume_json_path, jobs_csv_path):
    # Step 1: Extract text from resume
    if not os.path.exists(resume_json_path):
        raise FileNotFoundError(f"Resume file '{resume_json_path}' not found.")
    with open(resume_json_path, "r", encoding="utf-8") as file:
        try:
            resume_text = json.load(file)
            if isinstance(resume_text, str):
                resume_text=resume_text
            else:
                raise ValueError("Unexpected JSOn format: expected plain text.")
        except:
            raise ValueError("Invalid JSON file format.")

    if not resume_text.strip():
        raise ValueError("Parsed resume text is empty or invalid.")

    # Step 2: Load or generate embedding for the resume
    print("Loading or generating embedding for resume...")
    resume_embedding = load_resume_embedding()
    if not resume_embedding:
        resume_embedding = save_resume_embedding(resume_text)

    # Step 3: Load job listings
    print("Loading job listings...")
    jobs_df = load_job_listings(jobs_csv_path)

    # Step 4: Load existing embeddings and embed only new jobs
    print("Loading existing embeddings...")
    stored_embeddings = load_embeddings()

    print("Embedding new jobs...")
    new_embeddings = embed_new_jobs(jobs_df=jobs_df, stored_embeddings=stored_embeddings)

    # Update stored embeddings with new ones
    if new_embeddings:
        save_embeddings(new_embeddings)

    # Step 5: Rank job listings
    print("Ranking jobs...")
    updated_embeddings = {**stored_embeddings, **new_embeddings}
    ranked_jobs = rank_jobs(resume_embedding, updated_embeddings)

    return ranked_jobs


if __name__ == "__main__":
    resume_json_path = "parsed_resume.json"
    jobs_csv_path = "job_listings.csv"
    try:
        main(resume_json_path, jobs_csv_path)
    except Exception as e:
        print(f"Error: {e}")