
import streamlit as st
import pdfplumber
from docx import Document
from PIL import Image
import pytesseract
from transformers import pipeline
import spacy
from datetime import datetime

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        return "".join([page.extract_text() or "" for page in pdf.pages])

def extract_text_from_docx(file):
    doc = Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_txt(file):
    return file.read().decode('utf-8')

def extract_text_from_image(file):
    image = Image.open(file)
    return pytesseract.image_to_string(image)

def extract_keywords(text, top_n=5):
    doc = nlp(text)
    freq = {}
    for chunk in doc.noun_chunks:
        word = chunk.text.strip()
        freq[word] = freq.get(word, 0) + 1
    sorted_keywords = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [kw[0] for kw in sorted_keywords[:top_n]]

def generate_summary(text):
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
    return summary[0]['summary_text']

def generate_metadata(text, file_name, author=None):
    return {
        "title": file_name,
        "author": author or "Unknown",
        "date_created": datetime.now().strftime("%Y-%m-%d"),
        "keywords": extract_keywords(text),
        "summary": generate_summary(text),
        "document_type": "Unknown"
    }

st.set_page_config(page_title="Metadata Generator", layout="centered")

st.title("📄 Automated Metadata Generator")
st.write("Upload a PDF, DOCX, TXT, or Image file to extract metadata.")

uploaded_file = st.file_uploader("Upload Document", type=['pdf', 'docx', 'txt', 'jpg', 'jpeg', 'png'])

if uploaded_file:
    file_name = uploaded_file.name
    ext = file_name.split('.')[-1].lower()

    st.info(f"Processing `{file_name}` ...")

    try:
        if ext == 'pdf':
            text = extract_text_from_pdf(uploaded_file)
        elif ext == 'docx':
            text = extract_text_from_docx(uploaded_file)
        elif ext == 'txt':
            text = extract_text_from_txt(uploaded_file)
        elif ext in ['jpg', 'jpeg', 'png']:
            text = extract_text_from_image(uploaded_file)
        else:
            st.error("Unsupported file type.")
            st.stop()

        st.success("✅ Text extracted successfully!")
        st.subheader("📃 Extracted Text (first 1000 characters)")
        st.text_area("Text Preview", text[:1000], height=250)

        if st.button("Generate Metadata"):
            with st.spinner("Generating metadata..."):
                metadata = generate_metadata(text, file_name)
                st.subheader("📑 Generated Metadata")
                st.json(metadata)

    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
