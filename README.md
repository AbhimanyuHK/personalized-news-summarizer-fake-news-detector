# Personalized News Summarizer & Fake News Detector

An end-to-end NLP application that fetches online news articles, summarizes them using transformer models, and detects fake news using a trained classifier. Built using **Streamlit** for the frontend and **scikit-learn + spaCy** for NLP-based fake news detection.

---

## 🚀 Project Structure (Single Repo)

```
personalized-news-summarizer-fake-news-detector/
├── app.py                      # Streamlit app
├── train_nlp_classifier.py     # Train and save fake news classifier (.pkl)
├── requirements.txt            # Python dependencies
├── models/
│   └── fake_news_classifier.pkl
├── data/                       # Raw data: Fake.csv, True.csv
├── utils/
│   ├── summarizer.py
│   ├── classifier.py
│   └── news_fetcher.py
├── README.md                   # You are here
└── .gitignore
```

---

## 🧠 High-Level Design (HLD)

**Main Modules:**

* `User Preferences` (topics)
* `News Fetcher` (NewsAPI)
* `Summarizer` (transformers - BART/T5)
* `Fake News Classifier` (spaCy + TF-IDF + LogisticRegression)
* `Streamlit UI`

**Flow:**

```
User Input → Fetch News → Filter by Topics → Summarize → Classify → Show in UI
```

---

## 🛠 Development Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/personalized-news-summarizer-fake-news-detector
cd personalized-news-summarizer-fake-news-detector
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Train the Classifier (once)

```bash
python train_nlp_classifier.py
```

This creates `models/fake_news_classifier.pkl`

---

## 🧪 Run the App (Local Development)

```bash
streamlit run app.py
```

---

## 🧾 How to Use

1. Enter your news interests in the input (e.g., "AI, Economy")
2. Click "Fetch & Analyze"
3. See each article:

   * ✅ Title
   * ✂️ Summary (BART)
   * 📊 Credibility Score (% Real)

---

## 🌐 Deployment (Production Lifecycle)

### Deploy to Streamlit Cloud

1. Push your repo to GitHub
2. Go to [https://share.streamlit.io](https://share.streamlit.io)
3. Connect your repo → Select `app.py`

### Deploy to Heroku (Optional)

1. Add `Procfile`, `setup.sh`
2. Use `gunicorn` for backend + Streamlit

---

## 🧪 Model Lifecycle

| Stage           | Description                            |
| --------------- | -------------------------------------- |
| Data Collection | Use Fake.csv / True.csv                |
| Preprocessing   | spaCy: lemmatization, stopword removal |
| Feature Extract | TF-IDF (1-2 grams)                     |
| Classifier      | Logistic Regression (liblinear)        |
| Save            | `.pkl` with `joblib`                   |
| Load in App     | `classifier.py` module                 |

---

## 🤖 Tech Stack

| Layer      | Technology               |
| ---------- | ------------------------ |
| Frontend   | Streamlit                |
| NLP        | spaCy, scikit-learn      |
| Summarizer | HuggingFace Transformers |
| Model Save | `joblib`, `.pkl` format  |
| Hosting    | Streamlit Cloud / Heroku |

---

## 📌 Example Output

```
Title: Govt announces new AI policy
Summary: The Indian government introduced a comprehensive AI policy...
Credibility Score: 92%
Sentiment: Neutral
```

---

## ✨ Future Enhancements

* BERT/RoBERTa-based fake news classifier (via HuggingFace)
* Multilingual summarization
* Browser extension integration
* Real-time news alerts

---

## 🧑‍💻 Author

**Abhimanyu H. K.**
Email: [manyu1994@hotmail.com](mailto:manyu1994@hotmail.com)
