import streamlit as st
import joblib
from openai import OpenAI

# Configure OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=st.secrets["OPENROUTER_API_KEY"]
)

# Load SVM model and TF-IDF vectorizer
model = joblib.load('svm_fake_news_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Streamlit page setup
st.set_page_config(page_title="Fake News Detector + DeepSeek Fact Checker", layout="centered")
st.title("📰 Fake News Detector")
st.subheader("🚀 Classify Real vs Fake News & verify facts (DeepSeek R1)")

def verify_facts_with_openrouter(claim):
    prompt = f"""
You are a fact-checking assistant.

Claim: "{claim}"

Check the claim using publicly available facts.  
Return:
Verdict: Likely True or Likely False  
Reason: [Short explanation]
"""
    try:
        resp = client.chat.completions.create(
            model="deepseek/deepseek-r1:free",  # ✅ confirmed free :contentReference[oaicite:2]{index=2}
            messages=[
                {"role": "system", "content": "You are a helpful fact-checking assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"❌ Error during fact checking: {e}"

# User input
news_input = st.text_area("✍️ Enter news content:")

if st.button("🔍 Predict & Verify"):
    if not news_input.strip():
        st.warning("⚠️ Please enter some text.")
    else:
        # SVM prediction
        vec = vectorizer.transform([news_input])
        pred = model.predict(vec)[0]
        st.success("✅ Real News" if pred == 1 else "❌ Fake News")

        # Fact-check with LLM
        st.markdown("---")
        st.subheader("🧠 Fact Checker (DeepSeek R1)")
        with st.spinner("Checking facts..."):
            result = verify_facts_with_openrouter(news_input)
        st.markdown(result)
