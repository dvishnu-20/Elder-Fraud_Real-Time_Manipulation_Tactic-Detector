import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

def get_model_and_tokenizer(model_name="distilbert-base-uncased", num_labels=5):
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification"
    )
    return model, tokenizer
