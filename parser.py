import sys
import fitz
import openai
import os
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with fitz.open(pdf_path) as pdf:
            for page_num in range(len(pdf)):
                page = pdf.load_page(page_num)
                text += page.get_text()
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
    return text

# Prompt template for OpenAI API
prompt_template = """
Extract the following information in detail from the resume, and if any extra sections are present extract that also:

- Name:
- Email:
- Phone Number:
- LinkedIn URL:
- Skills:
- Work Experience:
- Education:
- Certifications:
- Projects:
- Publications:

Resume:
{}
"""

# Function to parse resume using OpenAI API
def parse_resume_with_openai(resume_text):
    prompt = prompt_template.format(resume_text)
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            temperature=0
        )
        if 'choices' in response and response['choices']:
            return response.choices[0].message['content'].strip()
        else:
            print("Unexpected API response format.")
            return None
    except Exception as e:
        print(f"Error during API call: {e}")

# Function to dynamically choose parsing mode
def parse_resume(pdf_path):
    print("Parsing resume locally using OpenAI...")
    resume_text = extract_text_from_pdf(pdf_path=pdf_path)
    return parse_resume_with_openai(resume_text=resume_text)

# Save parsed data to a file
def save_parsed_data(data, output_path="parsed_resume.json"):
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)
    print(f"Parsed data saved to {output_path}")

# Main execution
if __name__ == "__main__":
    pdf_path = sys.argv[1]
    parsed_resume = parse_resume(pdf_path=pdf_path)
    save_parsed_data(parsed_resume, output_path='parsed_resume.json')