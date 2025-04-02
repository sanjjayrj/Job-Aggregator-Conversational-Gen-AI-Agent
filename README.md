# Job Recommendation Conversational GenAI Agent

## Overview

This web application is a dynamic job recommendation platform where users can upload their resumes, set preferences, and get tailored job recommendations based on their skills and preferences. The platform uses OpenAI embeddings for matching jobs to resumes and integrates geographical proximity filtering to suggest relevant jobs near a user's specified location.

![alt text](https://github.com/sanjjayrj/Job-Aggregator-Conversational-Gen-AI-Agent/blob/main/diagram.png?raw=true)
---

## Features

1. **Resume Upload and Parsing**: Users can upload their resumes, which are parsed into plain text and stored in a JSON file for further processing.
2. **Job Scraping**: The application scrapes job listings based on user-defined job roles and locations, appending new jobs to a CSV file while avoiding duplicates.
3. **Job Embedding and Matching**:
   - Generates embeddings for parsed resumes and job descriptions using OpenAI's embedding models.
   - Matches jobs to resumes using cosine similarity.
4. **Geographical Proximity Filtering**:
   - Filters job recommendations based on their proximity to a user's specified location using geocoding and distance calculation.
5. **Real-Time Updates**:
   - Asynchronous scraping and matching to ensure real-time results.
   - The application dynamically refreshes to display new recommendations without requiring manual reloads.
6. **Chat Interface**:
   - A chatbot assistant to help users navigate the platform, provide career advice, and answer job-related questions.
7. **History Tracking**:
   - Automatically archives top job recommendations into a history section before replacing them with new recommendations.
8. **Markdown Rendering**:
   - Displays job details with rich formatting using markdown, enhancing readability.

---

## Tech Stack

1. **Frontend**:
   - HTML, CSS (for layout and styling)
   - JavaScript (for interactivity, chat interface, and AJAX requests)
2. **Backend**:
   - Flask (Python web framework)
   - Geopy (for geocoding and distance calculation)
   - Pandas (for data handling)
   - OpenAI (for embedding generation and chat completions)
3. **Database**:
   - CSV (to store job listings)
   - JSON (to store parsed resume data and embeddings)
4. **Third-Party APIs**:
   - OpenAI API (for embeddings and chatbot functionality)
   - Nominatim (OpenStreetMap API for geocoding)

---

## Installation

### Prerequisites

- Python 3.7 or above
- `pip` package manager
- OpenAI API Key (required for embeddings and chatbot functionality)

### Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/your-repo/job-recommendation-app.git
   cd job-recommendation-app
   ```

2. Create virtual environment:

    ```bash
    python -m venv dev
    ```

3. Install dependencies:

    ```bash
    pip install -r requirements.txt
    pip install -U python-jobspy
    ```

4. Set up environment variables:
    - Create a `.env` file in the project root directory:

    ```bash
    OPENAI_API_KEY=your_openai_api_key
    ```

5. Run the application:

    ```bash
    python app.py
    ```

---

## File Structure

- `app.py`: Main application file handling routes, job scraping, resume parsing, and chatbot integration
- `matcher.py`: Handles embedding generation for new jobs, loads old embeddings, matching and ranking of job listing to resume and preferences.
- `scraper.py`: Scrapes job listings from indeed, linkedin and glassdoor (currently only indeed is selected because of being rate liimited) based on user preferences.
- `parser.py`: Parses uploaded resume, and stores text data as well as uploaded resume in `uploads` folder.
- `static/`: Contains CSS, sample resumes and image assets.
- `templates/`: Contains HTML templates as well as Javascript within the HTML files.
- `job_listings.csv`: Created on running scraper and stores scraped job data.
- `parsed_resume.json`: Stores parsed resume text.
- `job_embeddings.json`: Stores embeddings for job descriptions.
- `top_jobs.json`: Stores the top job recommendations for the user.

---

## Features in Detail

**1. Chatbot Assistant:**
- Integrated with OpenAI's GPT to categorize intent of the user input into: update, general and unrelated, and acts accordingly.

- If update is triggered, the agent extracts the role and location the user is asking for and calls the scraper and matcher on another thread. And the job recommendations are updated below the chat.
- If the user is asking general job-related questions or details regarding user's resume or recommended jobs, the agent provides intelligent responses as it has access to this data.

**2. Geographical Filtering:**
- This bit is still a worrk in progress.
- Uses Nominatim for geocoding and Haversine formula to calculate distances.

- Filters jobs within a configurable radius (default: 15 miles) of the user's specified location.

### 3. Real-Time Updates

- Polls the backend for status updates to ensure that new job recommendations are displayed once scraping and matching are complete.

- Triggered when job updates in background.

### 4. Rich Job Details

* Displays job descriptions using Markdown for better readability.

---

## Known Issues and Improvements

* **Long Processing Times**: When scraping or embedding new jobs, there may be delays depending on the dataset size.

- **Fallback for Geocoding Errors**: This is still a work in progress. If geocoding fails for certain locations, jobs may not be filtered correctly.

- **Better Frontend Styling**: The chat interface and job history sections could be further improved for responsiveness and user experience.
- **Advanced Filtering**: Add additional filters (e.g., salary range, remote jobs) for better customization.

---

## Future Features

* **User Authentication**: Allow users to save preferences and history across sessions.
Advanced Matching: Incorporate skills and experience levels into the matching algorithm.
- **Job Alerts**: Email or notify users about new job recommendations.

- **Export Functionality**: Allow users to download job recommendations as a PDF or CSV.

---

## Contributions

Feel free to fork this repository and submit pull requests for improvements or new features. For major changes, please open an issue first to discuss your ideas.
