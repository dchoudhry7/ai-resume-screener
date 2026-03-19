import pdfplumber
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

nltk.download('stopwords')

model = SentenceTransformer('all-MiniLM-L6-v2')

skills_db = [
    "python", "machine learning", "data analysis",
    "sql", "flask", "django", "numpy", "pandas",
    "nlp", "deep learning", "excel"
]

role_skills = {
    "Data Analyst": ["sql", "excel", "python", "power bi", "data analysis"],
    "Data Scientist": ["python", "machine learning", "deep learning", "nlp", "statistics"],
    "Web Developer": ["html", "css", "javascript", "flask", "django"],
    "Software Engineer": ["python", "java", "data structures", "algorithms", "oop"],
    "ML Engineer": ["python", "machine learning", "deep learning", "tensorflow"],
    "Backend Developer": ["python", "django", "flask", "sql", "api"],
}

def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    words = text.split()
    words = [w for w in words if w not in stopwords.words('english')]
    return " ".join(words)

def tfidf_similarity(resume, jd):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume, jd])
    return cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

def semantic_similarity(resume, jd):
    embeddings = model.encode([resume, jd])
    return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

def extract_skills(text):
    return [skill for skill in skills_db if skill in text]

def compare_skills(resume_skills, jd_skills):
    matched = list(set(resume_skills) & set(jd_skills))
    missing = list(set(jd_skills) - set(resume_skills))
    return matched, missing

def calculate_score(tfidf, semantic, matched, jd_skills):
    skill_score = len(matched) / (len(jd_skills) + 1)
    final = (0.3 * tfidf + 0.5 * semantic + 0.2 * skill_score)
    return round(final * 100, 2)

def recommend_roles(resume_skills):
    role_scores = {}

    for role, skills in role_skills.items():
        match_count = len(set(resume_skills) & set(skills))
        score = match_count / len(skills)
        role_scores[role] = score

    # sort roles by score
    sorted_roles = sorted(role_scores.items(), key=lambda x: x[1], reverse=True)

    # return top 3 roles with decent match
    recommended = [role for role, score in sorted_roles if score > 0.3]

    return recommended[:3]

def generate_feedback(score, missing_skills):
    feedback = []

    if score > 75:
        feedback.append("Strong match! Your resume aligns well with the job.")
    elif score > 50:
        feedback.append("Moderate match. Improve some key skills.")
    else:
        feedback.append("Low match. You need to improve significantly.")

    if missing_skills:
        feedback.append("Consider adding these skills: " + ", ".join(missing_skills))

    if "project" not in missing_skills:
        feedback.append("Add more relevant projects to strengthen your profile.")

    return feedback