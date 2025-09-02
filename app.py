# app.py
import streamlit as st
from utils.news_fetcher import NewsFetcher
from utils.summarizer import NewsSummarizer
from utils.classifier import FakeNewsClassifier

# =========================
# Initialize Components
# =========================
st.set_page_config(page_title="News Summarizer & Fake News Detector", layout="wide")


@st.cache_resource
def load_components():
    fetcher = NewsFetcher()
    summarizer = NewsSummarizer()
    classifier = FakeNewsClassifier()
    return fetcher, summarizer, classifier


fetcher, summarizer, classifier = load_components()

# =========================
# UI Layout
# =========================
st.title("📰 Personalized News Summarizer & Fake News Detector")
st.markdown("Enter your **topic(s)** below and get summarized, credibility-checked news articles.")

query = st.text_input("🔍 Enter topic(s) of interest", "Artificial Intelligence")
num_articles = st.slider("Number of articles", 1, 10, 5)

if st.button("Fetch & Analyze"):
    with st.spinner("Fetching news..."):
        try:
            articles = fetcher.fetch(query, page_size=num_articles)
        except Exception as e:
            st.error(f"Error fetching news: {e}")
            articles = []

    if not articles:
        st.warning("No articles found. Try a different topic.")
    else:
        st.success(f"Fetched {len(articles)} articles")

        for i, art in enumerate(articles, 1):
            st.markdown(f"### {i}. {art['title']}")
            st.markdown(f"**Source:** {art['source']}  |  [Read more]({art['url']})")

            text_content = art["content"] or art["description"] or art["title"]

            # Summarization
            with st.spinner("Summarizing..."):
                summary = summarizer.summarize(text_content)

            st.markdown(f"✂️ **Summary:** {summary}")

            # Fake News Detection
            with st.spinner("Classifying..."):
                result = classifier.predict(text_content)

            st.markdown(
                f"📊 **Credibility:** {result['label']} "
                f"({result['score']}% confidence)"
            )

            st.markdown("---")
