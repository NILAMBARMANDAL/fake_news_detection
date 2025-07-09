# 📰 Fake News Detection App

This is a web application that detects whether a news article is **Fake** or **Real**, built using a **Support Vector Machine (SVM)** model trained on the WELFake dataset. The app is powered by **Streamlit** and deployed on the cloud for public use.

---

## 📊 Dataset

- **Name:** [WELFake Dataset](https://www.kaggle.com/datasets/nelgiriyewithana/welfake)
- **Source:** Kaggle
- **Size:** ~72000 articles
- **Labels:** 
  - `0`: Fake
  - `1`: Real

---

## 🤖 Model Details

- **Vectorizer:** TF-IDF (max features = 10,000)
- **Model Used:** SVM (Support Vector Machine)
- **Accuracy:** ~95.6%
- **Libraries Used:**
  - `scikit-learn`
  - `nltk`
  - `joblib`
  - `pandas`, `numpy`

---

## 🌐 Live App

> ✅ Click here to try the app:  
**[🔗 Open Fake News Detector](https://your-username-your-repo.streamlit.app)**  
*(Replace with your actual Streamlit app link)*

---

## 🛠 How to Run Locally

1. **Clone the repo**
```bash
git clone https://github.com/NILAMBARMANDAL/fake_news_detection.git
cd fake_news_detection
