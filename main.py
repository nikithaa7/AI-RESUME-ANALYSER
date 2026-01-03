import streamlit as st
from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
import re
from dotenv import load_dotenv
import os


load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("Groq API Key not found! Please add GROQ_API_KEY in your .env file.")
    st.stop() 


if "form_submitted" not in st.session_state:
    st.session_state.form_submitted = False
if "resume" not in st.session_state:
    st.session_state.resume = ""
if "job_desc" not in st.session_state:
    st.session_state.job_desc = ""


st.title("AI Resume Analyzer 📝")


def extract_pdf_text(uploaded_file):
    try:
        return extract_text(uploaded_file)
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return "Could not extract text from the PDF file."

def calculate_similarity_bert(text1, text2):
    ats_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    embeddings1 = ats_model.encode([text1])
    embeddings2 = ats_model.encode([text2])
    similarity = cosine_similarity(embeddings1, embeddings2)[0][0]
    return round(float(similarity), 3)

def get_report(resume, job_desc):
    try:
        client = Groq(api_key=api_key)
        prompt = f"""
# Context:
You are an AI Resume Analyzer.

# Instructions:
Analyze the resume based on the job description.
Score each point out of 5, using ✅ if aligned, ❌ if missing, ⚠️ if unclear.
Provide a "Suggestions to improve your resume:" section at the end.

# Inputs:
Candidate Resume: {resume}
Job Description: {job_desc}

# Output:
Each point should start with score and emoji, followed by explanation.
"""
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile"
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error generating report: {str(e)}"

def extract_scores(text):
    pattern = r'(\d+(?:\.\d+)?)/5'
    matches = re.findall(pattern, text)
    scores = [float(match) for match in matches]
    return scores


if not st.session_state.form_submitted:
    with st.form("resume_form"):
        resume_file = st.file_uploader("Upload your Resume/CV (PDF)", type="pdf")
        st.session_state.job_desc = st.text_area(
            "Enter Job Description:",
            placeholder="Job Description..."
        )
        submitted = st.form_submit_button("Analyze")
        
        if submitted:
            if resume_file and st.session_state.job_desc:
                st.info("Extracting Resume Text...")
                st.session_state.resume = extract_pdf_text(resume_file)
                st.session_state.form_submitted = True
                st.rerun()
            else:
                st.warning("Please provide both Resume and Job Description.")


if st.session_state.form_submitted:
    score_info = st.info("Calculating ATS similarity...")
    
  
    ats_score = calculate_similarity_bert(st.session_state.resume, st.session_state.job_desc)
    col1, col2 = st.columns(2)
    with col1:
        st.write("ATS Similarity Score (0-1):")
        st.subheader(str(ats_score))

 
    report = get_report(st.session_state.resume, st.session_state.job_desc)
    report_scores = extract_scores(report)
    
    avg_score = round(sum(report_scores) / len(report_scores), 2) if report_scores else 0.0
    with col2:
        st.write("AI Resume Report Average Score (out of 5):")
        st.subheader(str(avg_score))
    
    score_info.success("Scores generated successfully!")

  
    st.subheader("AI Generated Analysis Report:")
    st.text_area("Report:", report, height=500)
    
    st.download_button(
        label="Download Report",
        data=report,
        file_name="AI_Resume_Report.txt",
        icon="📥"
    )
