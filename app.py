import streamlit as st
import pandas as pd
import spacy
import re
import regex
import pdfplumber
import docx
from PIL import Image
import pytesseract
import os
import platform
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import matplotlib.pyplot as plt

# --- Tesseract Configuration ---
# On Windows, you might need to set the Tesseract path explicitly.
# 1. Find your Tesseract installation path (e.g., C:\Program Files\Tesseract-OCR\tesseract.exe)
# 2. Uncomment the line below and replace the path with your actual path.
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text(file):
    """Extracts text from an uploaded file based on its type."""
    text = ""
    file_name = file.name
    try:
        if file_name.endswith('.pdf'):
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""
        elif file_name.endswith('.docx'):
            doc = docx.Document(file)
            for para in doc.paragraphs:
                text += para.text + '\n'
        elif file_name.endswith(('.jpg', '.jpeg', '.png')):
            img = Image.open(file)
            text = pytesseract.image_to_string(img)
    except Exception as e:
        st.error(f"Error reading {file_name}: {e}")
        return None
    return text

def extract_email(text):
    match = re.search(r'[\w\.-]+@[\w\.-]+', text)
    return match.group(0) if match else 'N/A'

def extract_phone(text):
    match = re.search(r'(\+?\d{1,3}[\s-]?)?(\d{10})', text)
    return match.group(0) if match else 'N/A'

def extract_name(text):
    # Simple heuristic: first line or first capitalized phrase
    lines = text.splitlines()
    for line in lines:
        if line.strip() and len(line.split()) <= 5 and line[0].isupper():
            return line.strip()
    return 'N/A'

def extract_skills(text, skills_list):
    found_skills = set()
    for skill in skills_list:
        if re.search(r'\b' + re.escape(skill) + r'\b', text, re.IGNORECASE):
            found_skills.add(skill)
    return ', '.join(sorted(found_skills)) if found_skills else 'Not Found'
def extract_education(text):
    # Simple regex for degree and university
    degree_match = re.search(r'(Bachelor|Master|PhD|B\.Sc|M\.Sc|B\.Tech|M\.Tech|MBA|BBA|BCA|MCA)[^\n]*', text, re.IGNORECASE)
    university_match = re.search(r'(University|Institute|College|School)[^\n]*', text, re.IGNORECASE)
    degree = degree_match.group(0) if degree_match else 'Not Found'
    university = university_match.group(0) if university_match else 'Not Found'
    return degree, university

def extract_certifications(text):
    certs = re.findall(r'(Certified|Certification|Certificate)[^\n]*', text, re.IGNORECASE)
    return ', '.join(certs) if certs else 'Not Found'

def extract_projects(text):
    projects = re.findall(r'(Project|Projects)[^\n]*', text, re.IGNORECASE)
    return ', '.join(projects) if projects else 'Not Found'

def extract_experience(text):
    # Look for years of experience
    match = re.search(r'(\d+)\s+(?:years|yrs|year)\s+(?:of)?\s*experience', text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    # Fallback: look for job history
    exp_matches = re.findall(r'(\d{4})\s*-\s*(\d{4}|present)', text, re.IGNORECASE)
    if exp_matches:
        years = 0
        for start, end in exp_matches:
            end_year = int(end) if end.isdigit() else 2025
            years += end_year - int(start)
        return years
    return 0

def analyze_resume(resume_text, job_description, skills_list):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(resume_text)

    name = extract_name(resume_text)
    email = extract_email(resume_text)
    phone = extract_phone(resume_text)
    skills = extract_skills(resume_text, skills_list)
    experience = extract_experience(resume_text)
    degree, university = extract_education(resume_text)
    certifications = extract_certifications(resume_text)
    projects = extract_projects(resume_text)

    # Skill gap analysis
    job_skills = set([s.strip().lower() for s in skills_list])
    candidate_skills = set([s.strip().lower() for s in skills.split(',') if s != 'Not Found'])
    missing_skills = job_skills - candidate_skills

    # Calculate match score using TF-IDF and Cosine Similarity
    match_score = 0
    if resume_text and job_description:
        texts = [resume_text, job_description]
        try:
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(texts)
            match_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 100
        except ValueError:
            match_score = 0

    return {
        'Name': name,
        'Email': email,
        'Phone': phone,
        'Skills': skills,
        'Experience (years)': experience,
        'Degree': degree,
        'University': university,
        'Certifications': certifications,
        'Projects': projects,
        'Missing Skills': ', '.join(missing_skills) if missing_skills else 'None',
        'Match Score (%)': round(match_score, 2)
    }

# --- Main Streamlit App Logic ---
def main():
    st.set_page_config(page_title="Recruiter AI Resume Analyzer", layout="wide")

    # --- Theme Toggle ---
    theme = st.sidebar.selectbox("Choose Theme", ["Light", "Dark", "Blue"])
    if theme == "Dark":
        st.markdown("""
            <style>
            body, .stApp { background-color: #181818; color: #fff; }
            .stDataFrame { background-color: #222; }
            </style>
        """, unsafe_allow_html=True)
    elif theme == "Blue":
        st.markdown("""
            <style>
            body, .stApp { background: linear-gradient(90deg, #6a11cb 0%, #2575fc 100%); color: #fff; }
            .stDataFrame { background-color: #222; }
            </style>
        """, unsafe_allow_html=True)

    # --- Customizable Skills List ---
    st.sidebar.header("Skills List for Matching")
    default_skills = [
        'python', 'java', 'c++', 'sql', 'excel', 'machine learning', 'data science', 'deep learning',
        'nlp', 'react', 'node', 'django', 'flask', 'aws', 'azure', 'git', 'docker', 'kubernetes',
        'pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch', 'matplotlib', 'seaborn', 'power bi',
        'tableau', 'communication', 'leadership', 'problem solving', 'teamwork', 'project management'
    ]
    skills_input = st.sidebar.text_area("Edit/Add Skills (comma separated)", value=', '.join(default_skills), height=120)
    skills_list = [s.strip() for s in skills_input.split(',') if s.strip()]

    # --- UI Elements ---
    st.title("🧑‍💼 Recruiter AI: Multi-Resume Comparison & Best Candidate Selection")

    st.sidebar.header("Instructions")
    st.sidebar.info(
        """
        1. Enter the **Job Title** and **Job Requirements**.
        2. Upload multiple resumes (PDF, DOCX, JPG, PNG).
        3. Click **'Compare & Analyze'** to view tabular comparison and best candidate.
        """
    )

    st.header("1. Enter Job Details")
    job_title = st.text_input("Job Title", placeholder="e.g., Senior Python Developer")
    job_requirements = st.text_area("Job Requirements", placeholder="Paste or write the requirements here...", height=150)

    st.header("2. Upload Multiple Resumes (Drag & Drop Supported)")
    uploaded_files = st.file_uploader(
        "Upload resume files (PDF, DOCX, JPG, PNG)",
        type=['pdf', 'docx', 'jpg', 'jpeg', 'png'],
        accept_multiple_files=True,
        help="You can drag and drop multiple files here."
    )

    st.header("3. Compare & Analyze Candidates")
    progress_placeholder = st.empty()
    if st.button("Compare & Analyze", type="primary", use_container_width=True):
        if not job_requirements.strip() or not job_title.strip():
            st.error("Please provide both a Job Title and Job Requirements.")
        elif not uploaded_files:
            st.error("Please upload at least one resume.")
        else:
            results = []
            for idx, file in enumerate(uploaded_files):
                progress_placeholder.progress((idx+1)/len(uploaded_files), text=f"Processing {file.name}...")
                resume_text = extract_text(file)
                if resume_text:
                    analysis = analyze_resume(resume_text, job_requirements, skills_list)
                    analysis['File Name'] = file.name
                    # --- Keyword Highlighting ---
                    keywords = re.findall(r'\b\w{5,}\b', resume_text)
                    analysis['Keywords'] = ', '.join(keywords[:10]) if keywords else 'None'
                    results.append(analysis)
                else:
                    st.warning(f"Could not extract text from {file.name}. Skipping.")
            progress_placeholder.empty()

            if results:
                st.subheader("📊 Resume Comparison Table & Filters")
                column_order = ['File Name', 'Name', 'Match Score (%)', 'Experience (years)', 'Skills', 'Missing Skills', 'Degree', 'University', 'Certifications', 'Projects', 'Email', 'Phone', 'Keywords']
                results_df = pd.DataFrame(results).sort_values(by='Match Score (%)', ascending=False).reset_index(drop=True)

                # --- Advanced Filters ---
                min_exp = st.slider("Minimum Experience (years)", 0, int(results_df['Experience (years)'].max()), 0)
                skill_filter = st.text_input("Filter by Skill (comma separated)", "")
                filtered_df = results_df[results_df['Experience (years)'] >= min_exp]
                if skill_filter.strip():
                    skill_terms = [s.strip().lower() for s in skill_filter.split(',') if s.strip()]
                    filtered_df = filtered_df[filtered_df['Skills'].str.lower().apply(lambda x: any(term in x for term in skill_terms))]

                st.dataframe(filtered_df[column_order], use_container_width=True)

                # --- PDF/Excel Download ---
                st.download_button("Download Comparison Table (CSV)", filtered_df[column_order].to_csv(index=False), file_name="resume_comparison.csv", mime="text/csv")
                import io
                excel_buffer = io.BytesIO()
                filtered_df[column_order].to_excel(excel_buffer, index=False, engine='openpyxl')
                excel_buffer.seek(0)
                st.download_button("Download Comparison Table (Excel)", excel_buffer, file_name="resume_comparison.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

                # --- Skill Match Bar Chart ---
                st.subheader("Skill Match Percentage")
                plt.figure(figsize=(8,3))
                plt.bar(filtered_df['Name'], filtered_df['Match Score (%)'], color='#2575fc')
                plt.xlabel('Candidate')
                plt.ylabel('Match Score (%)')
                plt.xticks(rotation=45)
                st.pyplot(plt)

                # --- Experience Distribution Chart ---
                st.subheader("Experience Distribution")
                plt.figure(figsize=(8,3))
                plt.hist(filtered_df['Experience (years)'], bins=range(0, int(filtered_df['Experience (years)'].max())+2), color='#6a11cb', edgecolor='white')
                plt.xlabel('Years of Experience')
                plt.ylabel('Number of Candidates')
                st.pyplot(plt)

                # --- Candidate Profile Cards ---
                st.subheader("Candidate Profiles")
                for idx, row in filtered_df.iterrows():
                    st.markdown(f"""
<div style='background: linear-gradient(90deg, #2575fc 0%, #6a11cb 100%); color: #fff; padding: 16px; border-radius: 12px; margin-bottom: 16px;'>
    <h4 style='margin-top:0;margin-bottom:8px;'>{row['Name']} ({row['File Name']})</h4>
    <ul style='list-style:none;padding-left:0;'>
        <li><b>Match Score:</b> <span style='color:#ffe066;'>{row['Match Score (%)']}%</span></li>
        <li><b>Experience:</b> {row['Experience (years)']} years</li>
        <li><b>Skills:</b> {row['Skills']}</li>
        <li><b>Missing Skills:</b> {row['Missing Skills']}</li>
        <li><b>Degree:</b> {row['Degree']}</li>
        <li><b>University:</b> {row['University']}</li>
        <li><b>Certifications:</b> {row['Certifications']}</li>
        <li><b>Projects:</b> {row['Projects']}</li>
        <li><b>Email:</b> {row['Email']}</li>
        <li><b>Phone:</b> {row['Phone']}</li>
        <li><b>Keywords:</b> {row['Keywords']}</li>
    </ul>
</div>
""", unsafe_allow_html=True)

                # --- Candidate Notes Section (after all analysis) ---
                st.subheader("📝 Add Notes for Candidates")
                for idx, row in filtered_df.iterrows():
                    note_key = f"note_{row['File Name']}".replace(" ", "_")
                    note = st.text_area(f"Add notes for {row['Name']} ({row['File Name']})", key=note_key)

                # --- Best Candidate Highlight ---
                st.subheader("🏆 Best Candidate Recommendation")
                best_candidate = filtered_df.iloc[0]
                st.markdown(f"""
<div style='background: linear-gradient(90deg, #ff512f 0%, #dd2476 100%); color: #fff; padding: 20px; border-radius: 16px; box-shadow: 0 4px 16px rgba(0,0,0,0.12); margin-bottom: 24px;'>
    <h3 style='margin-top:0;margin-bottom:12px;'>Best Candidate: {best_candidate['Name']}</h3>
    <ul style='list-style:none;padding-left:0;'>
        <li><b>File:</b> {best_candidate['File Name']}</li>
        <li><b>Match Score:</b> <span style='color:#ffe066;'>{best_candidate['Match Score (%)']}%</span></li>
        <li><b>Experience:</b> {best_candidate['Experience (years)']} years</li>
        <li><b>Skills:</b> {best_candidate['Skills']}</li>
        <li><b>Missing Skills:</b> {best_candidate['Missing Skills']}</li>
        <li><b>Degree:</b> {best_candidate['Degree']}</li>
        <li><b>University:</b> {best_candidate['University']}</li>
        <li><b>Certifications:</b> {best_candidate['Certifications']}</li>
        <li><b>Projects:</b> {best_candidate['Projects']}</li>
        <li><b>Email:</b> {best_candidate['Email']}</li>
        <li><b>Phone:</b> {best_candidate['Phone']}</li>
        <li><b>Keywords:</b> {best_candidate['Keywords']}</li>
    </ul>
</div>
""", unsafe_allow_html=True)
                # --- Candidate Notes Section (after recommendation) ---

            else:
                st.error("Processing failed for all uploaded resumes.")

    st.caption("All processing is done locally and is free. No paid APIs or external services are used.")

    # --- Help/FAQ Section ---
    with st.expander("❓ Help / FAQ"):
        st.markdown("""
        **How to use this app:**
        - Enter job title and requirements.
        - Edit the skills list for matching if needed.
        - Upload multiple resumes (drag & drop supported).
        - Click 'Compare & Analyze' to view results.
        - Use filters to narrow down candidates.
        - Add notes for each candidate.
        - Download results as CSV/Excel.
        - All features are free and run locally.
        """)

    st.caption("All processing is done locally and is free. No paid APIs or external services are used.")

if __name__ == "__main__":
    main()