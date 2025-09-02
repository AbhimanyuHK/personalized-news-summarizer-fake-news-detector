# utils/classifier.py
import torch
from transformers import BertTokenizer, BertForSequenceClassification

class FakeNewsClassifier:
    def __init__(self, model_dir="models/bert_fake_news_model"):
        """
        Load fine-tuned BERT model for fake news detection.
        Args:
            model_dir (str): Path to saved model directory
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(model_dir)
        self.model = BertForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str) -> dict:
        """
        Predict if the given news text is Fake or True.
        Args:
            text (str): News article text
        Returns:
            dict: { 'label': 'Fake' or 'True', 'score': confidence (0-100%) }
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        label = "Fake" if probs[0] > probs[1] else "True"
        score = float(max(probs)) * 100

        return {"label": label, "score": round(score, 2)}


# Example usage
if __name__ == "__main__":
    clf = FakeNewsClassifier()
    sample_text = "NASA confirms earth will experience 15 days of darkness in November."
    prediction = clf.predict(sample_text)
    print("Prediction:", prediction)
