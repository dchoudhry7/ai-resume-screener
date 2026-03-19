from flask import Flask, render_template, request
from utils import *

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files['resume']
    jd = request.form['jd']

    resume_text = extract_text_from_pdf(file)

    resume_clean = preprocess(resume_text)
    jd_clean = preprocess(jd)

    tfidf = tfidf_similarity(resume_clean, jd_clean)
    semantic = semantic_similarity(resume_clean, jd_clean)

    resume_skills = extract_skills(resume_clean)
    jd_skills = extract_skills(jd_clean)

    matched, missing = compare_skills(resume_skills, jd_skills)

    score = calculate_score(tfidf, semantic, matched, jd_skills)

    recommended_roles = recommend_roles(resume_skills)

    feedback = generate_feedback(score, missing)

    return render_template(
        'result.html',
        score=score,
        matched=matched,
        missing=missing,
        feedback=feedback,
        tfidf=round(tfidf*100,2),
        semantic=round(semantic*100,2),
        recommended_roles=recommended_roles
    )

if __name__ == '__main__':
    app.run(debug=True)