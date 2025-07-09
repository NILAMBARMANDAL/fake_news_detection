import streamlit as st
import joblib
import wikipedia

# Load model and vectorizer
model = joblib.load('svm_fake_news_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Streamlit page config
st.set_page_config(page_title="Fake News Detector + Fact Checker", layout="centered")
st.title("📰 Fake News Detection App")
st.subheader("🚀 Classify news as Real or Fake and verify facts using Wikipedia")

# Input field
news_input = st.text_area("✍️ Enter the news content below:")

# Fact checking function
def verify_facts(text):
    try:
        # Select meaningful words only
        words = [word for word in text.split() if len(word) > 3 and word.isalpha()]
        checked = set()
        
        for word in words:
            keyword = word.lower()
            if keyword in checked:
                continue
            checked.add(keyword)
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

# Button action
if st.button("🔍 Predict & Verify"):
    if news_input.strip() == "":
        st.warning("⚠️ Please enter some news content.")
    else:
        # Predict using model
        vector_input = vectorizer.transform([news_input])
        prediction = model.predict(vector_input)[0]

        # Display prediction result
        if prediction == 1:
            st.success("✅ Prediction: Real News")
        else:
            st.error("❌ Prediction: Fake News")

        st.markdown("---")
        st.subheader("🧠 Wikipedia Fact Check")

        # Perform fact checking
        with st.spinner("Searching Wikipedia for relevant facts..."):
            fact_result = verify_facts(news_input)
        st.markdown(fact_result)
