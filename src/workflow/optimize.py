import json
import time
from groq import Groq
from src.config import GROQ_API_KEY
from src.utils.groq_retry import retry_groq_call


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

def call_groq_with_retry(messages, model="llama-3.3-70b-versatile", temperature=0.1, max_retries=5):
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not found in environment")

    # Set a 60-second timeout to prevent infinite hangs
    client = Groq(api_key=GROQ_API_KEY, timeout=60.0)

    def _call():
        response = client.chat.completions.create(
            messages=messages,
            model=model,
            temperature=temperature,
        )
        # Log successful API call
        try:
            from src.storage.tracker import increment_api_usage
            increment_api_usage()
        except Exception:
            pass
        return response.choices[0].message.content.strip()

    return retry_groq_call(_call, max_retries=max_retries)

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

def generate_cover_letter(cv_data, job_description):
    # Truncate to stay well within token limits
    job_desc_trimmed = (job_description or "")[:3000]

    prompt = f"""
    Write a job-specific professional cover letter based ONLY on the candidate CV and this job description.

    Requirements:
    - Maximum 280 words.
    - No fabricated experience, tools, projects, or claims.
    - Must map job requirements directly to existing CV evidence.
    - Professional business format:
      1) Salutation,
      2) Opening with role interest,
      3) 1-2 body paragraphs with role-specific fit,
      4) Short closing and signature line.
    - Keep tone professional, direct, and specific to this job description.
    - Avoid generic filler.

    CANDIDATE CV JSON:
    {json.dumps(cv_data, indent=2)}

    JOB DESCRIPTION:
    {job_desc_trimmed}

    Return ONLY the cover letter text. No preamble, no commentary.
    """

    try:
        return call_groq_with_retry(
            [{"role": "user", "content": prompt}],
            temperature=0.7,
            max_retries=6,
        )
    except Exception as e:
        print(f"Error generating cover letter: {e}")
        # Fallback: a shorter prompt (fewer tokens = more likely to clear a rate
        # limit), but still carrying enough CV + job specifics that the result
        # reads as tailored rather than generic boilerplate.
        try:
            name = cv_data.get("full_name", "the candidate")
            summary = cv_data.get("summary", "")
            skills = cv_data.get("skills", [])
            top_skills = ", ".join(str(s) for s in skills[:8]) if isinstance(skills, list) else ""
            experience = cv_data.get("experience", [])
            latest_role = ""
            if isinstance(experience, list) and experience:
                exp = experience[0]
                role = exp.get("role") or exp.get("title") or exp.get("job_title", "")
                company = exp.get("company") or exp.get("organization", "")
                if role or company:
                    latest_role = f"{role} at {company}".strip()
            simple_prompt = f"""Write a job-specific professional cover letter (max 200 words) for {name}.
Candidate summary: {summary}
Key skills: {top_skills}
Most recent role: {latest_role}
Job description (may be partial):
{job_desc_trimmed[:1800]}

Rules: reference specifics from the job description, map them to the candidate's skills/experience above,
no fabricated claims. Return only the cover letter text, no preamble."""
            return call_groq_with_retry(
                [{"role": "user", "content": simple_prompt}],
                temperature=0.7,
                max_retries=4,
            )
        except Exception as e2:
            print(f"Fallback cover letter also failed: {e2}")
            return ""


def render_tailored_cv_latex(cv_data):
    full_name = _latex_escape(cv_data.get("full_name", "Candidate Name"))
    location = _latex_escape(cv_data.get("location", "City, Country"))
    phone = _latex_escape(cv_data.get("phone", "+XX-XXXXXXXXXX"))
    email = _latex_escape(cv_data.get("email", "your@email.com"))

    links = cv_data.get("links", []) if isinstance(cv_data.get("links"), list) else []
    linkedin_link = next((l for l in links if "linkedin" in str(l).lower()), "https://linkedin.com")
    github_link = next((l for l in links if "github" in str(l).lower()), "https://github.com")

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
