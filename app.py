import streamlit as st
import joblib

# Load the saved model and vectorizer
model = joblib.load("svm_fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Title
st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detector")
st.markdown("Enter any news content below, and the model will predict whether it's **Real** or **Fake**.")

# User input
text_input = st.text_area("Paste the news article here:")

# Predict button
if st.button("Predict"):
    if not text_input.strip():
        st.warning("Please enter some text.")
    else:
        # Transform and predict
        transformed_text = vectorizer.transform([text_input])
        prediction = model.predict(transformed_text)[0]

        # Display result
        if prediction == 1:
            st.success("✅ This news is predicted to be **REAL**.")
        else:
            st.error("🚨 This news is predicted to be **FAKE**.")
