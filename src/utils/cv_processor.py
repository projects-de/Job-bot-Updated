import os
import json
import PyPDF2
from docx import Document
from groq import Groq
from src.config import GROQ_API_KEY
from src.utils.groq_retry import retry_groq_call

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        print(f"Error extracting PDF: {e}")
    return text

def extract_text_from_docx(docx_path):
    text = ""
    try:
        doc = Document(docx_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        print(f"Error extracting DOCX: {e}")
    return text

def parse_cv_with_ai(cv_text):
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not found in environment")
    
    client = Groq(api_key=GROQ_API_KEY)
    
    prompt = f"""
    You are a professional HR assistant. Parse the following CV text into a structured JSON format.
    The JSON should contain:
    - full_name
    - email
    - phone
    - location
    - links (LinkedIn, GitHub, Portfolio, etc.)
    - summary (a short professional summary)
    - skills (a list of technical and soft skills)
    - experience (a list of objects with title, company, duration, and bullet points)
    - education (a list of objects with degree, institution, and year)
    - projects (a list of objects with title and description)

    Return ONLY the raw JSON.

    CV TEXT:
    {(cv_text or "")[:6000]}
    """
    
    def _call(model):
        return client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=0.1,
        )

    response = retry_groq_call(
        lambda: _call("llama-3.3-70b-versatile"),
        fallback_fn=lambda: _call("llama-3.1-8b-instant"),
    )

    try:
        text = response.choices[0].message.content.strip()
        # Handle cases where model adds text before/after JSON
        if "{" in text and "}" in text:
            text = text[text.find("{"):text.rfind("}")+1]
        
        parsed = json.loads(text)
        return parsed
    except Exception as e:
        print(f"Error parsing JSON from Groq: {e}")
        print(f"Raw response: {response.choices[0].message.content}")
        return None

def process_cv(file_path):
    _, ext = os.path.splitext(file_path)
    if ext.lower() == '.pdf':
        text = extract_text_from_pdf(file_path)
    elif ext.lower() in ['.doc', '.docx']:
        text = extract_text_from_docx(file_path)
    else:
        raise ValueError("Unsupported file format")
    
    return parse_cv_with_ai(text)
