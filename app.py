import streamlit as st
import joblib
import re
import nltk
import sys
import os
from nltk.corpus import stopwords
from langdetect import detect, LangDetectException
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from datetime import datetime
import pandas as pd



nltk.download('stopwords', quiet=True)

def custom_tokenizer(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    words = text.split()
    stop_words = set(stopwords.words('english'))
    return [w for w in words if w not in stop_words and len(w) > 2]

sys.modules['__main__'].custom_tokenizer = custom_tokenizer

if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=['text_snippet', 'prediction', 'confidence', 'timestamp', 'language'])

def detect_content_language(text):
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"

def generate_pdf_report(prediction, confidence, indicators):
    filename = "news_report.pdf"
    c = canvas.Canvas(filename, pagesize=letter)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 750, "News Authenticity Report")
    c.line(100, 745, 500, 745)
    c.setFont("Helvetica", 12)
    y_position = 700
    c.drawString(100, y_position, f"Prediction: {'Fake News' if prediction else 'Real News'}")
    y_position -= 30
    c.drawString(100, y_position, f"Confidence: {confidence:.1f}%")
    y_position -= 30
    c.drawString(100, y_position, "Key Indicators:")
    y_position -= 15
    for idx, word in enumerate(indicators[:5], 1):
        c.drawString(120, y_position, f"{idx}. {word}")
        y_position -= 20
    c.drawString(100, 50, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    c.save()
    return filename

def apply_dark_mode():
    if st.session_state.get('dark_mode', False):
        st.markdown("""
        <style>
            .reportview-container {background: #0E1117; color: white;}
            .stTextArea textarea {color: white;}
        </style>
        """, unsafe_allow_html=True)

st.set_page_config(page_title="News Validator", layout="wide")

with st.sidebar:
    st.header("Display Settings")
    dark_mode = st.toggle("Dark Mode", key='dark_mode')
    apply_dark_mode()

st.title("📰 Advanced News Validator")
user_input = st.text_area("Paste news content here:", height=300)


try:
    vectorizer = joblib.load('models/vectorizer.pkl')
    model = joblib.load('models/model.pkl')
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

if st.button("Analyze"):
    if len(user_input) < 100:
        st.warning("Please enter at least 100 characters")
    else:
        lang = detect_content_language(user_input)
        if lang != 'en':
            st.warning(f"Detected language: {lang.upper()}")
        cleaned = ' '.join(custom_tokenizer(user_input))
        features = vectorizer.transform([cleaned])
        pred = model.predict(features)[0]
        proba = model.predict_proba(features)[0][pred] * 100
        indicators = vectorizer.get_feature_names_out()
        new_entry = pd.DataFrame([{
            'text_snippet': user_input[:100],
            'prediction': 'Fake' if pred else 'Real',
            'confidence': proba,
            'timestamp': datetime.now(),
            'language': lang
        }])
        st.session_state.history = pd.concat([st.session_state.history, new_entry])
        if pred:
            st.error(f"🚩 Fake News ({proba:.1f}% confidence)")
        else:
            st.success(f"✅ Real News ({proba:.1f}% confidence)")
        pdf_path = generate_pdf_report(pred, proba, indicators)
        with open(pdf_path, "rb") as f:
            st.download_button("Download PDF Report", f, file_name="news_analysis.pdf", mime="application/pdf")
        csv = st.session_state.history.to_csv(index=False)
        st.download_button("Export History as CSV", csv, "analysis_history.csv", "text/csv")

with st.expander("View Analysis History"):
    st.dataframe(st.session_state.history, height=300)
