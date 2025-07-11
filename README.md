# 📰 Fake News Detection App + Fact Checker (LLM-Powered)

A web-based app that detects whether a news article is **Fake** or **Real** using a trained **SVM model**, and then performs **fact-checking** using a **free LLM** (`DeepSeek R1`) via OpenRouter.

---

## 📦 Dataset

- **Name:** [WELFake Dataset](https://www.kaggle.com/datasets/nelgiriyewithana/welfake)
- **Size:** ~72,000 articles
- **Labels:**
  - `0` = Fake
  - `1` = Real

---

## ⚙️ Model Details

| Component     | Description                         |
|---------------|-------------------------------------|
| Model         | Support Vector Machine (SVM)        |
| Vectorizer    | TF-IDF (max_features=10,000)        |
| Accuracy      | ~95.6%                              |

---

## 🌐 Live App

> ✅ Try it live:  
**[🔗 Open Fake News Detector](https://your-username-your-repo.streamlit.app)**  
*(Replace with your actual Streamlit Cloud URL)*

---

## 🔍 Features

- ✅ Classifies news as Real or Fake using ML
- 🧠 Fact-checks claims using LLM (DeepSeek R1)
- 🪶 Uses lightweight model via OpenRouter (Free API)
- 🚫 No billing or credit card required

---

## 🛠 How to Run Locally

### 1. Clone the Repo

```bash
git clone https://github.com/your-username/fake_news_detection.git
cd fake_news_detection
