import streamlit as st
import joblib
import wikipedia

# Load the trained model and TF-IDF vectorizer
model = joblib.load('svm_fake_news_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

st.set_page_config(page_title="Fake News Detector + Fact Checker", layout="centered")

st.title("📰 Fake News Detection App")
st.subheader("🚀 Classify news as Real or Fake and verify facts using Wikipedia")

# User input
news_input = st.text_area("✍️ Enter the news content below:")

# Prediction
if st.button("Predict"):
    if news_input.strip() == "":
        st.warning("Please enter some news content.")
    else:
        input_transformed = vectorizer.transform([news_input])
        prediction = model.predict(input_transformed)

        label = "✅ Real News" if prediction[0] == 1 else "❌ Fake News"
        st.markdown(f"### Prediction: {label}")

# Fact checking with Wikipedia
def verify_facts(text):
    try:
        # Filter out short/common words and focus on capitalized or meaningful ones
        words = [word for word in text.split() if len(word) > 3 and word[0].isalpha()]
        checked = set()

        for word in words:
            word_lower = word.lower()
            if word_lower in checked:
                continue
            checked.add(word_lower)

            try:
                summary = wikipedia.summary(word, sentences=2)
                if word.lower() in summary.lower():
                    return f"🧠 Found info on **{word}**:\n\n> {summary}"
            except wikipedia.exceptions.DisambiguationError:
                continue
            except wikipedia.exceptions.PageError:
                continue

        return "⚠️ No matching fact found on Wikipedia."
    except Exception as e:
        return f"❌ Error during fact checking: {str(e)}"
