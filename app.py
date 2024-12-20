from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
from parser import parse_resume, save_parsed_data
from scraper import main as scrape_jobs
from matcher import main as match_jobs, save_resume_embedding
import os
import openai
import re
import markdown
import json
import threading
import pandas as pd

app = Flask(__name__)

openai.api_key = os.getenv("OPENAI_API_KEY")

# Configure file upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

parsed_resume_file = "parsed_resume.json"
scraped_jobs_file = "job_listings.csv"
top_jobs_file = "top_jobs.json"
CHAT_HISTORY_FILE = "chat_history.json"


# Status flags
status = {
    "scraping_complete": False,
    "matching_complete": False
}

# Job History
job_history = []

@app.route('/')
def upload_page():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'resume' not in request.files:
        return "No file uploaded", 400
    
    resume = request.files['resume']
    if resume.filename == '':
        return "No selected file", 400
    
    # Save the uploaded resume
    resume_path = os.path.join(UPLOAD_FOLDER, 'resume.pdf')
    resume.save(resume_path)

    # Parse the resume
    parsed_data = parse_resume(resume_path)
    save_parsed_data(parsed_data, parsed_resume_file)
    return redirect(url_for('preferences_page'))

@app.route('/preferences', methods=['GET', 'POST'])
def preferences_page():
    if request.method == 'POST':
        job_location = request.form['job_location']
        job_role = request.form['job_role']

        # Reset status flags
        status["scraping_complete"] = False
        status["matching_complete"] = False

        # Run scraper and matcher in the background
        def scrape_and_match():
            try:
                # Scrape jobs
                scrape_jobs(search_term=job_role, location=job_location)
                status["scraping_complete"] = True

                # Run matcher to find top jobs
                if os.path.exists(scraped_jobs_file):
                    ranked_jobs = match_jobs(resume_json_path=parsed_resume_file, jobs_csv_path=scraped_jobs_file)
                    if ranked_jobs is not None:
                        filtered_jobs = ranked_jobs[ranked_jobs['location'].str.contains(job_location, case=False, na=False)]
                        top_jobs = filtered_jobs.head(10).to_dict(orient='records')
                        pd.DataFrame(top_jobs).to_json(top_jobs_file, orient="records")
                    status["matching_complete"] = True
            except Exception as e:
                print(f"Error in scrape_and_match: {e}")
    
        threading.Thread(target=scrape_and_match).start()

        return redirect(url_for('dashboard'))

    return render_template('preferences.html')

@app.route('/dashboard')
def dashboard():
    global job_history
    resume_path = 'resume.pdf'
    job_history = remove_duplicates(job_history)
    chat_history = load_chat_history()

    return render_template('dashboard.html', jobs=[], job_history=job_history, resume_path=resume_path, chat_history=chat_history)

@app.route('/status', methods=['GET'])
def get_status():
    # API to check if scraping and matching are complete
    return jsonify(status)


@app.route('/get_top_jobs', methods=['GET'])
def get_top_jobs():
    try:
        if status["matching_complete"] and os.path.exists(top_jobs_file):
            with open(top_jobs_file, "r") as file:
                top_jobs = json.load(file)
            
            # Convert Markdown descriptions to HTML
            for job in top_jobs:
                job['description'] = markdown.markdown(job['description'])

            print("Top jobs data:", top_jobs)
            return jsonify({"jobs": top_jobs})
        else:
            print("Top jobs not ready or file missing.")  # Debug log
            return jsonify({"jobs": []})
    except Exception as e:
        print(f"Error in /get_top_jobs: {e}")
        return jsonify({"error": "Unable to fetch top jobs"}), 500

# Job History
def remove_duplicates(job_history):
    seen = set()  # To store unique (title, company) combinations
    unique_jobs = []
    for job in job_history:
        identifier = (job['title'], job['company'])  # Create a tuple of title and company
        if identifier not in seen:
            seen.add(identifier)
            unique_jobs.append(job)
    return unique_jobs

@app.route('/save_job', methods=['POST'])
def save_job():
    job = request.json
    job_history.append(job)
    job_history = remove_duplicates(job_history)
    return {"message": "Job saved to history!"}, 200

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(file_path):
        print(f"Serving file: {file_path}")
        return send_from_directory(UPLOAD_FOLDER, filename)
    print(f"File not found: {file_path}")
    return "File not found", 404

@app.route('/replace_resume', methods=['POST'])
def replace_resume():
    if 'new_resume' not in request.files:
        return "No file uploaded", 400
    
    new_resume = request.files['new_resume']
    if new_resume.filename == '':
        return "No selected file", 400
    
    # Replace the existing resume
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'resume.pdf')
    new_resume.save(filepath)

    # Parse the new resume
    parsed_data = parse_resume(filepath)
    save_parsed_data(parsed_data, parsed_resume_file)

    # Regenerate the embedding for the new resume
    try:
        with open(parsed_resume_file, "r") as file:
            parsed_resume_text = json.load(file)
        print("Generating new resume embedding...")
        save_resume_embedding(parsed_resume_text)  # Save the updated resume embedding
    except Exception as e:
        print(f"Error generating resume embedding: {e}")
        return "Error updating resume embedding", 500

    # Move current top jobs to job history
    if os.path.exists(top_jobs_file):
        with open(top_jobs_file, "r") as file:
            current_top_jobs = json.load(file)
        global job_history
        job_history.extend(current_top_jobs)  # Append current top jobs to history
        job_history = remove_duplicates(job_history)

    # Reset matching status
    status["matching_complete"] = False

    # Rerun matcher with existing scraped jobs
    def rerun_matcher():
        try:
            if os.path.exists(scraped_jobs_file):
                ranked_jobs = match_jobs(resume_json_path=parsed_resume_file, jobs_csv_path=scraped_jobs_file)
                if ranked_jobs is not None:
                    top_jobs = ranked_jobs.head(10).to_dict(orient='records')
                    pd.DataFrame(top_jobs).to_json(top_jobs_file, orient="records")
                status["matching_complete"] = True
        except Exception as e:
            print(f"Error in rerun_matcher: {e}")

    threading.Thread(target=rerun_matcher).start()

    return redirect(url_for('dashboard'))


PROMPT_TEMPLATE = """
You are a job recommendation assistant. You can do the following:
1. Answer questions related to job opportunities, career advice, and job descriptions based on the parsed resume and current recommendations.
2. If the user specifies a new location or job role, update the search parameters and run the scraper and matcher in the background.

Here is the parsed resume data: {resume_data}

Here are the top jobs recommendations: {top_jobs}

Instructions:
- Use the `parsed resume data` to answer general career questions and provide advice tailored to the user's profile.
- Use the `top job recommendations` to answer specific questions about these jobs, such as job descriptions, company details, or location.
- If the user asks for a new "location" or "job role" explicitly, acknowledge the update, run the scraper asynchronously, and update the recommendations.

Examples:
- If the user asks about a specific job in the recommendations, provide details about that job.
- If the user asks about opportunities in a new location or role, acknowledge and start the update process.

When responding, be concise, friendly and professional.
"""

def load_chat_history():
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, "r") as file:
            return json.load(file)
    return []

def save_chat_history(chat_history):
    with open(CHAT_HISTORY_FILE, "w") as file:
        json.dump(chat_history, file)

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.form['message']  # User's input from the chat window
    chat_history = load_chat_history()
    
    # Add user message to chat history
    chat_history.append({"sender": "user", "message": user_message})

    # Load parsed resume data
    with open(parsed_resume_file, "r") as file:
        resume_data = json.load(file)
    # Load top job recommendations
    if os.path.exists(top_jobs_file):
        with open(top_jobs_file, "r") as file:
            top_jobs = json.load(file)
    else:
        top_jobs = []

    # Convert Markdown in job descriptions to HTML
    for job in top_jobs:
        job['description'] = markdown.markdown(job['description'])
    # Create the prompt
    prompt = PROMPT_TEMPLATE.format(
        resume_data=json.dumps(resume_data, indent=2),
        top_jobs=json.dumps(top_jobs, indent=2)
    )

    # Append the user message to the conversation
    full_prompt = f"{prompt}\nUser: {user_message}\nAssistant:"

    try:
        # Determine intent using GPT
        intent = detect_update_intent(user_message, openai.api_key)

        assistant_reply = ""

        if intent == "update":
            # Extract job role and location (basic example; refine for accuracy)
            job_role, location = extract_job_and_location(user_message)
            threading.Thread(target=update_scraper_and_matcher, args=(job_role, location)).start()
            assistant_reply = (
                f"I will update the search to focus on {job_role} jobs in {location}. "
                "I will provide you with relevant job recommendations below shortly."
            )

        elif intent == "general":
            # Respond to general job-related queries using GPT
            full_prompt = f"{prompt}\nUser: {user_message}\nAssistant:"
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",  # or "gpt-4"
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=300,
                temperature=0.7
            )
            assistant_reply = response['choices'][0]['message']['content']
        else:
            # Handle unrelated queries
            assistant_reply = "I'm here to help with job-related questions. Let me know how I can assist you."
        
        # Add assistant reply to chat history
        chat_history.append({"sender": "assistant", "message": assistant_reply})
        save_chat_history(chat_history)

        return jsonify({"message": assistant_reply})

    except Exception as e:
        print(f"Error in chat: {e}")
        return jsonify({"error": "Unable to process the chat request"}), 500


def detect_update_intent(user_message, openai_api_key):
    openai.api_key = openai_api_key

    prompt = """
        You are an intelligent assistant that determines user intent based on the following user message:
        - If the user mentions moving to a new location or looking for specific jobs, return "update".
        - If the user is asking about job-related information or what resume changes the user should make but not requesting job updates, return "general".
        - If the user's message does not relate to jobs, return "unrelated".

        Message: "{}"
        Response (one of: update, general, unrelated):""".format(user_message)

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # or "gpt-4"
            messages=[{"role": "system", "content": prompt}],
            max_tokens=10,
            temperature=0
        )
        intent = response['choices'][0]['message']['content'].strip().lower()
        return intent
    except Exception as e:
        print(f"Error in detecting intent: {e}")
        return "error"

def extract_job_and_location(user_message):
    # Use GPT to extract job role and location
    prompt = f"""
                You are a smart assistant that extracts a "job role" and a "location" from user messages.
                User Message: "{user_message}"
                If the user doesn't mention a job role, return "default" for job role.
                If the user doesn't mention a location, return "default" for location.
                The output location should be of format: Manhattan, NY
                Output format: Job Role: <job_role>, Location: <location>
            """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # or "gpt-4"
            messages=[{"role": "system", "content": prompt}],
            max_tokens=50,
            temperature=0.5
        )
        content = response['choices'][0]['message']['content'].strip()
        # Parse job role and location from response
        lines = content.split(",")
        job_role = lines[0].split(":")[1].strip() if "Job Role:" in lines[0] else "default"
        location = lines[1].split(":")[1].strip() if "Location:" in lines[1] else "default"
        return job_role, location
    except Exception as e:
        print(f"Error extracting job and location: {e}")
        return "default", "default"

def update_scraper_and_matcher(job_role, location):
    try:
        # Reset status flags
        status["scraping_complete"] = False
        status["matching_complete"] = False

        # Run scraper and matcher in the background
        def scrape_and_match():
            global job_history
            try:
                # Scrape jobs
                print(f"Scraping jobs for role: {job_role} in location: {location}")
                scrape_jobs(search_term=job_role, location=location)
                status["scraping_complete"] = True

                # Run matcher to find top jobs
                if os.path.exists(scraped_jobs_file):
                    ranked_jobs = match_jobs(resume_json_path=parsed_resume_file, jobs_csv_path=scraped_jobs_file)
                    print("The ranked jobs are: ", ranked_jobs.head(5))

                    if ranked_jobs is not None:
                        # Filter top 10 jobs from the requested location
                        filtered_jobs = ranked_jobs[ranked_jobs['location'].str.contains(location, case=False, na=False)]
                        print("The filtered jobs are: ", filtered_jobs.head(5))
                        top_jobs = filtered_jobs.head(10).to_dict(orient='records')

                        archive_old_top_jobs()
                        save_top_jobs(top_jobs=top_jobs)
                        print("Top jobs updated and old jobs archived.")

                    status["matching_complete"] = True
            except Exception as e:
                print(f"Error in scrape_and_match: {e}")

        threading.Thread(target=scrape_and_match).start()
    except Exception as e:
        print(f"Error in update_scraper_and_matcher: {e}")

# Helper Functions
def archive_old_top_jobs():
    """Move old top jobs into job history."""
    global job_history
    if os.path.exists(top_jobs_file):
        with open(top_jobs_file, "r") as file:
            old_top_jobs = json.load(file)
            try:
                print(f"Archiving {len(old_top_jobs)} jobs to job history.")
                job_history.extend(old_top_jobs)
                job_history = remove_duplicates(job_history)  # Remove duplicates
            except Exception as e:
                print(f"Error archiving old top jobs: {e}")

def save_top_jobs(top_jobs):
    """Save new top jobs to the top jobs file."""
    try:
        print(f"Saving {len(top_jobs)} new top jobs.")
        pd.DataFrame(top_jobs).to_json(top_jobs_file, orient="records")
    except Exception as e:
        print(f"Error saving top jobs: {e}")

if __name__ == '__main__':
    app.run(debug=True)
