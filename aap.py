import streamlit as st
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
import re

st.set_page_config(page_title="Offline Study-Buddy & Question Predictor", layout="wide")

st.title("📚 Offline AI Study-Buddy & Exam Question Predictor")
st.caption("100% Private | Runs Locally on CPU | Zero Cloud Dependency")

uploaded_file = st.file_uploader("Apne Class Notes ya Syllabus ki PDF upload karo", type=["pdf"])

def extract_text(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text

def get_key_topics(text, top_n=8):
    clean_text = re.sub(r'[^a-zA-Z\s]', '', text)
    sentences = [s.strip() for s in clean_text.split('\n') if len(s.strip()) > 10]
    
    if not sentences:
        return []
    
    vectorizer = TfidfVectorizer(stop_words='english', max_features=top_n)
    tfidf_matrix = vectorizer.fit_transform(sentences)
    feature_names = vectorizer.get_feature_names_out()
    return feature_names

if uploaded_file is not None:
    st.success("File Successfully Uploaded!")
    
    with st.spinner("Processing Document Locally..."):
        text_content = extract_text(uploaded_file)
        
        st.subheader("📌 Key Topics & Predicted Exam Concepts")
        keywords = get_key_topics(text_content)
        
        cols = st.columns(2)
        for i, kw in enumerate(keywords):
            cols[i % 2].info(f"🔹 **Important Topic {i+1}:** {kw.capitalize()}")
            
        st.divider()
        st.subheader("📝 Practice Question Generator")
        st.write("In topics par expected exam questions:")
        for kw in keywords:
            st.write(f"- What is **{kw.capitalize()}**? Explain its core principles and application.")

else:
    st.info("Kripya koi PDF file upload karke check karein.")