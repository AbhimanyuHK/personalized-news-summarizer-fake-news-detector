# Personalized News Summarizer & Fake News Detector

An end-to-end NLP application that fetches online news articles, summarizes them using transformer models, and detects fake news using a **BERT-based classifier**. Built using **Streamlit** for the frontend and **HuggingFace Transformers** for both summarization and classification.

---

## 🚀 Project Structure (Single Repo)

```
personalized-news-summarizer-fake-news-detector/
├── app.py                      # Streamlit app
├── train_bert_classifier.py    # Fine-tune and save BERT fake news classifier
├── requirements.txt            # Python dependencies
├── models/
│   └── bert_fake_news_model/   # Saved BERT model directory
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
* `Fake News Classifier` (BERT-based fine-tuned model)
* `Streamlit UI`

**Flow:**

```
User Input → Fetch News → Filter by Topics → Summarize → BERT Classification → Show in UI
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
```

### 3. Train or Load the BERT Classifier

If using pre-trained model:

```bash
# Ensure models/bert_fake_news_model/ contains config.json, pytorch_model.bin, tokenizer
```

To train your own:

```bash
python train_bert_classifier.py
```

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
   * 📊 Credibility Score (based on BERT)

---

## 🌐 Deployment (Production Lifecycle)

### Deploy to Streamlit Cloud

1. Push your repo to GitHub
2. Go to [https://share.streamlit.io](https://share.streamlit.io)
3. Connect your repo → Select `app.py`

### Deploy to HuggingFace Space (Optional)

1. Upload model to HuggingFace Hub
2. Use `pipeline()` with model path

---

## 🧪 Model Lifecycle

| Stage           | Description                             |
| --------------- | --------------------------------------- |
| Data Collection | Use Fake.csv / True.csv                 |
| Tokenization    | BERT tokenizer from HuggingFace         |
| Fine-tuning     | BERT base model + binary classification |
| Save Model      | HuggingFace model.save\_pretrained()    |
| Load in App     | via `pipeline('text-classification')`   |

---

## 🤖 Tech Stack

| Layer      | Technology               |
| ---------- | ------------------------ |
| Frontend   | Streamlit                |
| NLP        | HuggingFace Transformers |
| Summarizer | BART / T5                |
| Classifier | BERT fine-tuned          |
| Hosting    | Streamlit Cloud / HF Hub |

---

## 📌 Example Output

```
Title: Govt announces new AI policy
Summary: The Indian government introduced a comprehensive AI policy...
Credibility Score: 96%
Sentiment: Neutral
```

---

## ✨ Future Enhancements

* DistilBERT for faster classification
* Fact-checking via web scraping or Wikipedia
* Browser extension integration
* Re-ranking based on user feedback

---

## 🧑‍💻 Author

**Abhimanyu H. K.**
Email: [manyu1994@hotmail.com](mailto:manyu1994@hotmail.com)
