# utils/summarizer.py
from transformers import pipeline


class NewsSummarizer:
    def __init__(self, model_name="facebook/bart-large-cnn"):
        """
        Initialize summarization pipeline.
        Default model: BART Large CNN (good for news summarization).
        """
        self.summarizer = pipeline("summarization", model=model_name)

    def summarize(self, text, max_len=130, min_len=30):
        """
        Generate a summary for the given text.

        Args:
            text (str): Full news article text
            max_len (int): Max length of summary tokens
            min_len (int): Min length of summary tokens

        Returns:
            str: Concise summary
        """
        if not text or len(text.split()) < 40:
            # Skip too-short articles
            return text

        summary = self.summarizer(
            text,
            max_length=max_len,
            min_length=min_len,
            do_sample=False
        )
        return summary[0]["summary_text"]


# Example usage
if __name__ == "__main__":
    sample_text = """
    The Indian government has introduced a new artificial intelligence policy 
    that aims to boost AI research, innovation, and deployment in critical sectors 
    like healthcare, education, and agriculture. The policy outlines funding, 
    academic collaboration, and international partnerships to ensure India becomes 
    a global AI leader.
    """

    summarizer = NewsSummarizer()
    print("Summary:", summarizer.summarize(sample_text))
