# train_bert_classifier.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch

# =============================
# 1. Load Data
# =============================
fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")

# Label data
fake["label"] = 0  # Fake
true["label"] = 1  # True

df = pd.concat([fake, true]).sample(frac=1, random_state=42).reset_index(drop=True)

# =============================
# 2. Train-Test Split
# =============================
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["text"].tolist(),
    df["label"].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# =============================
# 3. Tokenizer
# =============================
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)


train_dataset = Dataset.from_dict({"text": train_texts, "labels": train_labels})
test_dataset = Dataset.from_dict({"text": test_texts, "labels": test_labels})

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

# Keep only necessary columns
train_dataset = train_dataset.remove_columns(["text"])
test_dataset = test_dataset.remove_columns(["text"])

train_dataset.set_format("torch")
test_dataset.set_format("torch")

# =============================
# 4. Model
# =============================
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# =============================
# 5. Training Arguments
# =============================
training_args = TrainingArguments(
    output_dir="./bert_fake_news_results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)


# =============================
# 6. Compute Metrics
# =============================
def compute_metrics(pred):
    logits, labels = pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    report = classification_report(labels, preds, target_names=["Fake", "True"], output_dict=True)
    return {"accuracy": acc, "precision": report["weighted avg"]["precision"],
            "recall": report["weighted avg"]["recall"], "f1": report["weighted avg"]["f1-score"]}


# =============================
# 7. Trainer
# =============================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer
)

# =============================
# 8. Train
# =============================
trainer.train()

# =============================
# 9. Save Model
# =============================
save_dir = "models/bert_fake_news_model"
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

print(f"✅ Model saved to {save_dir}")
