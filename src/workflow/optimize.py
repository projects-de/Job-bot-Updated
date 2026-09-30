import json
import time
from datetime import datetime
from groq import Groq
from src.config import GROQ_API_KEY
from src.utils.groq_retry import retry_groq_call


def _normalize_links(cv_data):
    """
    cv_data["links"] is stored as a dict (e.g. {"LinkedIn": "linkedin.com/in/..."})
    in the real profile JSON, but older code here assumed a list and silently
    dropped every link when it got a dict instead. Normalize to a flat list of
    strings so both shapes work.
    """
    links = cv_data.get("links", [])
    if isinstance(links, dict):
        return [str(v) for v in links.values() if v]
    if isinstance(links, list):
        return [str(v) for v in links if v]
    return []


def _latex_escape(value):
    if value is None:
        return ""
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _pick_text(item, keys):
    if not isinstance(item, dict):
        return ""
    for key in keys:
        val = item.get(key)
        if val:
            return str(val)
    return ""


def _normalize_bullets(value):
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        lines = [v.strip("- ").strip() for v in value.split("\n") if v.strip()]
        return lines if lines else [value.strip()]
    return []

def call_groq_with_retry(messages, model="openai/gpt-oss-120b", temperature=0.1, max_retries=5, fallback_model="openai/gpt-oss-20b"):
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not found in environment")

    # Set a 60-second timeout to prevent infinite hangs
    client = Groq(api_key=GROQ_API_KEY, timeout=60.0)

    def _call(m):
        response = client.chat.completions.create(
            messages=messages,
            model=m,
            temperature=temperature,
            # Same fix as cv_processor.py: the gpt-oss models spend part of
            # their output budget on a hidden reasoning trace before the real
            # answer, which can truncate the actual content on a long prompt
            # if no explicit budget is given — "low" effort favors the answer
            # over reasoning depth (this is extraction/generation, not a task
            # that benefits from deep reasoning), and 4000 tokens comfortably
            # covers the longest output this call ever produces (a full
            # tailored CV JSON or a cover letter).
            reasoning_effort="low",
            max_tokens=4000,
        )
        # Log successful API call
        try:
            from src.storage.tracker import increment_api_usage
            increment_api_usage()
        except Exception:
            pass
        return response.choices[0].message.content.strip()

    fallback_fn = (lambda: _call(fallback_model)) if fallback_model and fallback_model != model else None
    return retry_groq_call(lambda: _call(model), max_retries=max_retries, fallback_fn=fallback_fn)

def tailor_cv(cv_data, job_description):
    prompt = f"""
    You are an expert ATS resume optimizer.

    Task:
    Use the uploaded CV JSON and THIS job description to produce a stronger, job-matched CV JSON.
    Before rewriting, identify the role requirements and match them to existing CV evidence.
    Then rewrite summary/experience/projects to highlight only matching evidence from the CV.

    Non-negotiable rules:
    1. Keep the EXACT same top-level JSON keys and nested schema from the input CV JSON.
    2. Do NOT invent any skill, tool, company, project, education item, metric, or date.
    3. Reorder and rephrase only to improve relevance to the job description.
    4. Prefer concise ATS-friendly wording with job-relevant keywords when they are already supported by the CV.
    5. If a requirement is not supported by CV evidence, do not claim it.
    6. Ensure summary clearly states fit for this specific role.
    7. Ensure experience bullets prioritize impact and relevance to this specific role.

    ORIGINAL CV JSON:
    {json.dumps(cv_data, indent=2)}

    JOB DESCRIPTION:
    {job_description}

    Output rules:
    - Return ONLY valid JSON.
    - No markdown, no commentary, no code fences.
    """
    
    try:
        text = call_groq_with_retry([{"role": "user", "content": prompt}], temperature=0.1)
        
        # Extract JSON if model adds fluff
        if "{" in text and "}" in text:
            text = text[text.find("{"):text.rfind("}")+1]
            
        return json.loads(text)
    except Exception as e:
        print(f"Error tailoring CV: {e}")
        return cv_data  # Fallback to original

def _lean_cv_for_prompt(cv_data, max_experience=2, max_bullets=2, max_skills=10):
    """
    Trim the CV down to only what an LLM prompt actually needs (name, summary,
    skills, recent experience highlights, education) — the full CV JSON runs
    ~1,300 tokens on its own, which matters on Groq's token quota when several
    jobs get processed back-to-back. Defaults are deliberately lean since most
    callers (form-field answering) only need facts, not narrative depth;
    generate_cover_letter below passes richer explicit limits since it's the
    one place where more CV detail directly improves output quality.
    """
    experience = cv_data.get("experience", [])
    lean_experience = []
    if isinstance(experience, list):
        for exp in experience[:max_experience]:
            role = exp.get("role") or exp.get("title") or exp.get("job_title", "")
            company = exp.get("company") or exp.get("organization", "")
            duration = exp.get("duration") or exp.get("dates") or exp.get("period", "")
            desc = exp.get("description") or exp.get("bullet_points") or exp.get("highlights")
            if isinstance(desc, list):
                bullets = [str(b) for b in desc[:max_bullets]]
            elif isinstance(desc, str):
                bullets = [b.strip("-•* ").strip() for b in desc.split("\n") if b.strip()][:max_bullets]
            else:
                bullets = []
            lean_experience.append({"role": role, "company": company, "duration": duration, "highlights": bullets})

    education = cv_data.get("education", [])
    lean_education = []
    if isinstance(education, list):
        for edu in education[:2]:
            lean_education.append({
                "degree": edu.get("degree") or edu.get("title", ""),
                "institution": edu.get("institution") or edu.get("school", ""),
            })

    skills = cv_data.get("skills", [])
    return {
        "full_name": cv_data.get("full_name", ""),
        # Cheap scalar facts (a few tokens each) — kept even in the lean version because
        # form-field answering genuinely needs them, e.g. a "Verify/Confirm Email" field
        # can only be answered correctly if the LLM knows what email was already entered.
        "email": cv_data.get("email", ""),
        "phone": cv_data.get("phone", ""),
        "location": cv_data.get("location", ""),
        # Included so a "LinkedIn/GitHub/Portfolio URL" form field gets answered
        # with the real link instead of the model inventing a plausible-looking one.
        "links": _normalize_links(cv_data),
        "summary": cv_data.get("summary", ""),
        "skills": skills[:max_skills] if isinstance(skills, list) else skills,
        "experience": lean_experience,
        "education": lean_education,
    }


def _cover_letter_header(cv_data: dict) -> str:
    """
    Name / contact line / date block. Built deterministically in code rather
    than asked of the LLM — contact details must be copied exactly (a model
    can silently mangle a phone number or pick the wrong link), and "today's
    date" is a fact the LLM has no reliable way to know, not something worth
    risking a hallucination on.
    """
    name = str(cv_data.get("full_name") or "").strip()
    phone = str(cv_data.get("phone") or "").strip()
    email = str(cv_data.get("email") or "").strip()
    location = str(cv_data.get("location") or "").strip()

    links = _normalize_links(cv_data)
    linkedin_url = next((l.strip() for l in links if "linkedin" in l.lower()), "")

    contact_line = " | ".join(p for p in (phone, email, location, linkedin_url) if p)
    date_line = datetime.now().strftime("%B %d, %Y")

    return "\n".join(line for line in (name, contact_line, date_line) if line)


def _assemble_cover_letter(cv_data: dict, body: str) -> str:
    """Wrap an LLM-written (or template-written) body with the deterministic header/salutation/signature."""
    name = str(cv_data.get("full_name") or "").strip() or "the candidate"
    header = _cover_letter_header(cv_data)
    body = (body or "").strip()
    parts = [header, "Dear Hiring Manager,", body, "Sincerely,", name]
    return "\n\n".join(p for p in parts if p)


def _cover_letter_body_prompt(lean_cv: dict, job_desc_trimmed: str, max_words: int) -> str:
    return f"""
    Write ONLY the body paragraphs of a job-specific professional cover letter — no header, no
    date, no "Dear Hiring Manager", no "Sincerely" or signature, those are added separately.
    Base it strictly on the candidate CV JSON and this job description.

    Structure (plain paragraphs, no markdown, no bullet points, no headings):
    1. Opening paragraph — express interest in the specific role from the job description, and
       give a one-sentence high-level value proposition (years of experience + domain fit).
    2. One or two body paragraphs — describe concrete achievements from the candidate's actual
       experience (real companies, titles, technologies, and metrics already present in the CV
       JSON) and map them directly to what this job description is asking for.
    3. Closing paragraph — reinforce fit and enthusiasm, and note availability to discuss further.

    Non-negotiable rules:
    - {max_words} words maximum.
    - Every fact must come from the CANDIDATE CV JSON below — no fabricated companies, titles,
      technologies, dates, or metrics. If the CV doesn't support a claim, don't make it.
    - Do not include a salutation, header, date, or signature — start directly with the opening
      paragraph's first sentence and end after the closing paragraph.
    - Professional, direct tone. No generic filler ("I am a hard worker", "I am a team player").

    CANDIDATE CV JSON:
    {json.dumps(lean_cv, separators=(',', ':'))}

    JOB DESCRIPTION:
    {job_desc_trimmed}

    Return ONLY the paragraphs of the letter body, separated by blank lines. No preamble, no
    commentary, no markdown.
    """


def _deterministic_fallback_body(cv_data: dict, job_description: str) -> str:
    """
    Last-resort, no-LLM cover letter body — used only if every Groq attempt
    (including the fallback model in retry_groq_call) fails outright. Still
    built entirely from the candidate's real CV data, never a generic
    unrelated placeholder, so a cover letter is always produced and uploaded
    rather than silently skipped.
    """
    summary = str(cv_data.get("summary") or "").strip()
    skills = cv_data.get("skills", [])
    top_skills = ", ".join(str(s) for s in skills[:8]) if isinstance(skills, list) else ""
    experience = cv_data.get("experience", [])
    latest = experience[0] if isinstance(experience, list) and experience else {}
    role = latest.get("role") or latest.get("title") or latest.get("job_title", "")
    company = latest.get("company") or latest.get("organization", "")

    paragraphs = []
    opening = "I am writing to express my interest in this role."
    if summary:
        opening += f" {summary}"
    paragraphs.append(opening)

    if role or company:
        where = f"{role or 'my current role'}{f' at {company}' if company else ''}"
        paragraphs.append(
            f"In my experience as {where}, I have built hands-on expertise with "
            f"{top_skills or 'the core skills this role requires'}, which I believe maps "
            "directly to what you are looking for in this position."
        )

    paragraphs.append(
        "I would welcome the opportunity to discuss how my experience can support your team's "
        "goals. Thank you for considering my application."
    )
    return "\n\n".join(paragraphs)


def generate_cover_letter(cv_data, job_description):
    # Truncate to stay well within token limits. The CV side of this prompt is
    # deliberately kept lean (see _lean_cv_for_prompt) so this richer job-description
    # allowance and the fuller letter structure below don't blow the token budget.
    job_desc_trimmed = (job_description or "")[:2600]
    lean_cv = _lean_cv_for_prompt(cv_data, max_experience=3, max_bullets=4, max_skills=15)

    try:
        body = call_groq_with_retry(
            [{"role": "user", "content": _cover_letter_body_prompt(lean_cv, job_desc_trimmed, max_words=380)}],
            temperature=0.7,
            max_retries=4,
        )
        return _assemble_cover_letter(cv_data, body)
    except Exception as e:
        print(f"Error generating cover letter: {e}")
        # Fallback: a leaner CV context (fewer tokens = more likely to clear a rate
        # limit), still real CV facts, shorter target length.
        try:
            lean_cv_short = _lean_cv_for_prompt(cv_data, max_experience=1, max_bullets=2, max_skills=8)
            body = call_groq_with_retry(
                [{"role": "user", "content": _cover_letter_body_prompt(lean_cv_short, job_desc_trimmed[:1800], max_words=220)}],
                temperature=0.7,
                max_retries=3,
            )
            return _assemble_cover_letter(cv_data, body)
        except Exception as e2:
            print(f"Fallback cover letter generation also failed: {e2}")
            # Never return an empty cover letter — a deterministic, CV-grounded
            # template guarantees a real, system-generated letter always gets
            # uploaded instead of the upload step being silently skipped.
            return _assemble_cover_letter(cv_data, _deterministic_fallback_body(cv_data, job_description))


def render_tailored_cv_latex(cv_data):
    full_name = _latex_escape(cv_data.get("full_name", "Candidate Name"))
    location = _latex_escape(cv_data.get("location", "City, Country"))
    phone = _latex_escape(cv_data.get("phone", "+XX-XXXXXXXXXX"))
    email = _latex_escape(cv_data.get("email", "your@email.com"))

    links = _normalize_links(cv_data)
    linkedin_link = next((l for l in links if "linkedin" in l.lower()), "https://linkedin.com")
    github_link = next((l for l in links if "github" in l.lower()), "https://github.com")

    summary = _latex_escape(cv_data.get("summary", "Professional summary tailored to this role."))

    education = cv_data.get("education", []) if isinstance(cv_data.get("education"), list) else []
    experience = cv_data.get("experience", []) if isinstance(cv_data.get("experience"), list) else []
    projects = cv_data.get("projects", []) if isinstance(cv_data.get("projects"), list) else []
    certifications = cv_data.get("certifications", [])
    achievements = cv_data.get("achievements", [])
    skills = cv_data.get("skills", [])

    if not isinstance(certifications, list):
        certifications = [certifications] if certifications else []
    if not isinstance(achievements, list):
        achievements = [achievements] if achievements else []
    if not isinstance(skills, list):
        skills = [skills] if skills else []

    lines = [
        "%%%%",
        "% Job Tailored Resume",
        "%%%%",
        "",
        r"\documentclass[letterpaper,10pt]{article}",
        "",
        r"\usepackage[empty]{fullpage}",
        r"\usepackage{titlesec}",
        r"\usepackage{enumitem}",
        r"\usepackage[hidelinks]{hyperref}",
        r"\usepackage{fancyhdr}",
        r"\usepackage{fontawesome5}",
        r"\usepackage{multicol}",
        r"\usepackage{bookmark}",
        r"\usepackage{lastpage}",
        r"\usepackage{charter}",
        r"\usepackage{xcolor}",
        "",
        r"\definecolor{accentTitle}{HTML}{0e6e55}",
        r"\definecolor{accentText}{HTML}{0e6e55}",
        r"\definecolor{accentLine}{HTML}{a16f0b}",
        "",
        r"\pagestyle{fancy}",
        r"\fancyhf{}",
        r"\fancyfoot{}",
        r"\renewcommand{\headrulewidth}{0pt}",
        r"\renewcommand{\footrulewidth}{0pt}",
        "",
        r"\urlstyle{same}",
        "",
        r"\addtolength{\oddsidemargin}{-0.65in}",
        r"\addtolength{\evensidemargin}{-0.65in}",
        r"\addtolength{\textwidth}{1.3in}",
        r"\addtolength{\topmargin}{-0.8in}",
        r"\addtolength{\textheight}{1.6in}",
        "",
        r"\setlength{\multicolsep}{-2pt}",
        r"\setlength{\columnsep}{-1pt}",
        r"\setlength{\tabcolsep}{0pt}",
        r"\setlength{\footskip}{3.7pt}",
        r"\raggedbottom",
        r"\raggedright",
        "",
        r"\newcommand{\documentTitle}[2]{",
        r"  \begin{center}",
        r"    {\Huge\color{accentTitle} #1}\\[6pt]",
        r"    {\small #2}\\[3pt]",
        r"    {\color{accentLine}\hrule}",
        r"  \end{center}",
        r"}",
        "",
        r"\titleformat{\section}{",
        r"  \vspace{-4pt}",
        r"  \color{accentText}",
        r"  \raggedright\large\bfseries",
        r"}{}{0em}{}[\color{accentLine}\titlerule]",
        "",
        r"\newcommand{\heading}[2]{",
        r"  \textbf{#1} \hfill \textit{#2}\\",
        r"}",
        "",
        r"\newenvironment{resume_list}{",
        r"  \vspace{-5pt}",
        r"  \begin{itemize}[itemsep=-2pt, parsep=0pt, leftmargin=20pt]",
        r"}{",
        r"  \end{itemize}",
        r"}",
        "",
        r"\renewcommand\labelitemi{--}",
        "",
        r"\begin{document}",
        "",
        r"\documentTitle{" + full_name + "}{",
        "   " + location + r" \\",
        r"   {\small ",
        r"   \faPhone\ " + phone + r" ~|~ ",
        r"   \faEnvelope\ \href{mailto:" + email + "}{" + email + r"} ~|~",
        r"   \faLinkedin\ \href{" + _latex_escape(linkedin_link) + r"}{LinkedIn} ~|~",
        r"   \faGithub\ \href{" + _latex_escape(github_link) + r"}{GitHub}",
        r"   }",
        r"}",
        "",
        r"\section{Summary}",
        summary,
        "",
        r"\section{Education}",
    ]

    if education:
        for edu in education[:3]:
            degree = _latex_escape(_pick_text(edu, ["degree", "title", "program"]))
            institution = _latex_escape(_pick_text(edu, ["institution", "school", "university"]))
            year = _latex_escape(_pick_text(edu, ["year", "duration", "dates"]))
            city = _latex_escape(_pick_text(edu, ["location", "city"]))
            lines.append(r"\heading{" + (institution or "University Name") + "}{" + (year or "Year -- Year") + "}")
            lines.append(r"\textit{" + (degree or "Degree Title") + r"} \hfill " + (city or "City, Country"))
    else:
        lines.extend([
            r"\heading{University Name}{Year -- Year}",
            r"\textit{Degree Title} \hfill City, Country",
        ])

    lines.extend(["", r"\section{Work Experience}"])

    if experience:
        for exp in experience[:4]:
            company = _latex_escape(_pick_text(exp, ["company", "organization", "employer"]))
            duration = _latex_escape(_pick_text(exp, ["duration", "dates", "period"]))
            role = _latex_escape(_pick_text(exp, ["role", "title", "job_title", "position"]))
            desc_val = exp.get("description") or exp.get("bullet_points") or exp.get("highlights")
            bullets = _normalize_bullets(desc_val)[:4]

            lines.append(r"\heading{" + (company or "Company Name") + "}{" + (duration or "Start Date -- End Date") + "}")
            lines.append(r"\textbf{" + (role or "Job Title") + "}")
            lines.append(r"\begin{resume_list}")
            if bullets:
                for bullet in bullets:
                    lines.append(r"  \item " + _latex_escape(bullet))
            else:
                lines.append(r"  \item Describe responsibility or achievement with measurable impact.")
            lines.append(r"\end{resume_list}")
    else:
        lines.extend([
            r"\heading{Company Name}{Start Date -- End Date}",
            r"\textbf{Job Title}",
            r"\begin{resume_list}",
            r"  \item Describe responsibility or achievement.",
            r"\end{resume_list}",
        ])

    lines.extend(["", r"\section{Technical Skills}"])
    skills_joined = ", ".join(_latex_escape(s) for s in skills[:20]) if skills else "List languages and tools"
    lines.extend([
        r"\begin{itemize}[itemsep=2pt, parsep=0pt]",
        r"  \item \textbf{Programming:} " + skills_joined,
        r"  \item \textbf{Frameworks/Tools:} " + skills_joined,
        r"  \item \textbf{Databases:} " + skills_joined,
        r"  \item \textbf{Other:} Cloud, DevOps, collaboration",
        r"\end{itemize}",
        "",
        r"\section{Projects}",
    ])

    if projects:
        for proj in projects[:3]:
            title = _latex_escape(_pick_text(proj, ["title", "name"]))
            date = _latex_escape(_pick_text(proj, ["date", "duration", "year"]))
            desc = _normalize_bullets(proj.get("description"))[:3]
            lines.append(r"\heading{" + (title or "Project Title") + "}{" + (date or "Date") + "}")
            lines.append(r"\begin{resume_list}")
            if desc:
                for bullet in desc:
                    lines.append(r"  \item " + _latex_escape(bullet))
            else:
                lines.append(r"  \item Brief description of project and technologies used.")
            lines.append(r"\end{resume_list}")
    else:
        lines.extend([
            r"\heading{Project Title}{Date}",
            r"\begin{resume_list}",
            r"  \item Brief project description and impact.",
            r"\end{resume_list}",
        ])

    lines.extend(["", r"\section{Certifications}", r"\begin{resume_list}"])
    if certifications:
        for cert in certifications[:5]:
            lines.append(r"  \item " + _latex_escape(cert))
    else:
        lines.append(r"  \item Certification Name --- Organization")
    lines.append(r"\end{resume_list}")

    lines.extend(["", r"\section{Achievements}", r"\begin{resume_list}"])
    if achievements:
        for ach in achievements[:5]:
            lines.append(r"  \item " + _latex_escape(ach))
    else:
        lines.append(r"  \item Awards, recognitions, or measurable wins.")
    lines.extend([r"\end{resume_list}", "", r"\end{document}"])

    return "\n".join(lines)
