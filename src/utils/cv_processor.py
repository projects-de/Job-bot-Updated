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

def _extract_json_object(raw_text: str) -> str:
    """Strip markdown code fences (```json ... ``` or ``` ... ```) and any
    leading/trailing prose the model added despite being told not to, leaving
    just the {...} object for json.loads."""
    text = (raw_text or "").strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
    if "{" in text and "}" in text:
        text = text[text.find("{"):text.rfind("}") + 1]
    return text.strip()


def parse_cv_with_ai(cv_text, filename_hint: str = ""):
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not found in environment")

    client = Groq(api_key=GROQ_API_KEY)

    # Some PDFs (confirmed live: a resume with a stylized bold name header)
    # extract with zero space characters between letters — e.g. a real name
    # "Muhammad Faiz Ali Faizi" comes out of PyPDF2 as "MUHAMMADFAIZALIFAIZI".
    # The model then has to guess word boundaries with no real signal and can
    # guess wrong ("Muhammad Faizal Ifaizi"). The uploaded filename is a
    # second, independent, properly-spaced data point — people overwhelmingly
    # name resume files after themselves — so pass it as a hint the model can
    # cross-reference; it costs nothing when the filename is generic.
    filename_hint_block = (
        f'\n    The file was uploaded with the filename "{filename_hint}" — if the CV '
        f"text above has the candidate's name mashed together without spaces or "
        f"garbled, use this filename (minus any words like resume/cv/job titles) "
        f"to help resolve the correct spelling and spacing of their name.\n"
        if filename_hint else ""
    )

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
{filename_hint_block}
    Return ONLY the raw JSON.

    CV TEXT:
    {(cv_text or "")[:6000]}
    """

    def _call(model):
        return client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=0.1,
            # gpt-oss models spend part of their output budget on a hidden
            # reasoning trace before the actual answer; a long, detailed CV
            # extraction can exhaust that budget and get its JSON truncated
            # mid-object (finish_reason "length") — 8000 was confirmed live to
            # comfortably fit a dense multi-page CV's structured output, and
            # "low" reasoning effort leaves more of that budget for the answer
            # itself since this is a straight extraction task, not one that
            # benefits from deep reasoning.
            reasoning_effort="low",
            max_tokens=8000,
        )

    # Two full attempts, primary model then fallback — not just on a network
    # rate-limit (retry_groq_call's own fallback trigger) but also when the
    # model *did* respond successfully yet produced JSON that doesn't parse
    # (confirmed live: intermittent on a long resume, not deterministic — a
    # bare "return None" here used to give up on the very first bad parse
    # instead of giving the fallback model a chance).
    last_error = None
    last_raw = None
    for model, fallback_model in (("openai/gpt-oss-120b", "openai/gpt-oss-20b"), ("openai/gpt-oss-20b", None)):
        response = retry_groq_call(
            lambda m=model: _call(m),
            fallback_fn=(lambda fm=fallback_model: _call(fm)) if fallback_model else None,
        )
        choice = response.choices[0]
        raw_text = choice.message.content or ""
        last_raw = raw_text
        if choice.finish_reason == "length":
            last_error = "Response was truncated (hit the token limit) before completing the JSON"
            print(f"[WARN] CV parse on {model}: {last_error}")
            continue
        try:
            return json.loads(_extract_json_object(raw_text))
        except Exception as e:
            last_error = str(e)
            print(f"[WARN] CV parse on {model}: JSON decode failed: {e}")
            print(f"Raw response: {raw_text[:2000]}")
            continue

    raise ValueError(
        f"LLM returned unparsable output after trying both models — last error: {last_error}. "
        f"Raw response (truncated): {(last_raw or '')[:500]}"
    )

def process_cv(file_path):
    stem, ext = os.path.splitext(file_path)
    if ext.lower() == '.pdf':
        text = extract_text_from_pdf(file_path)
    elif ext.lower() in ['.doc', '.docx']:
        text = extract_text_from_docx(file_path)
    else:
        raise ValueError("Unsupported file format")

    filename_hint = os.path.basename(stem).replace("_", " ").replace("-", " ").strip()
    return parse_cv_with_ai(text, filename_hint=filename_hint)
