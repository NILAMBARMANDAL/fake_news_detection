import streamlit as st
import joblib

import google.generativeai as genai

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
# Load model and vectorizer
model = joblib.load('svm_fake_news_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Configure Streamlit page
st.set_page_config(page_title="Fake News Detector + Gemini Fact Checker", layout="centered")
st.title("📰 Fake News Detection App")
st.subheader("🚀 Classify news as Real or Fake and verify facts using Gemini LLM")

# Input field
news_input = st.text_area("✍️ Enter the news content below:")

# --- Gemini Config ---
GEMINI_API_KEY = "YOUR_API_KEY_HERE"  # Replace with your actual key
genai.configure(api_key=GEMINI_API_KEY)

model_gemini = genai.GenerativeModel('gemini-pro')

# Gemini-based Fact Checking Function
def verify_facts_with_gemini(claim):
    prompt = f"""
You are a fact-checking assistant.

Claim: "{claim}"

Check the claim using publicly available facts and common knowledge.
Return your answer in this format:
Verdict: Likely True or Likely False
Reason: [Short explanation]
"""
    try:
        response = model_gemini.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ Error during Gemini fact checking: {str(e)}"

# Button click
if st.button("🔍 Predict & Verify"):
    if news_input.strip() == "":
        st.warning("⚠️ Please enter some news content.")
    else:
        # Predict
        vector_input = vectorizer.transform([news_input])
        prediction = model.predict(vector_input)[0]

        if prediction == 1:
            st.success("✅ Prediction: Real News")
        else:
            st.error("❌ Prediction: Fake News")

        st.markdown("---")
        st.subheader("🧠 Gemini Fact Checker")

        # Fact check with Gemini
        with st.spinner("Verifying facts using Gemini..."):
            fact_result = verify_facts_with_gemini(news_input)
        st.markdown(fact_result)
