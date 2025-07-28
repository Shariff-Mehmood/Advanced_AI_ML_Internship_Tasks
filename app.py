import streamlit as st
from transformers import BertTokenizerFast, BertForSequenceClassification
import torch
import numpy as np

# Load model and tokenizer
model_path = "bert_news_model"
tokenizer = BertTokenizerFast.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# Set model to evaluation mode
model.eval()

# AG News Labels
labels = ["World", "Sports", "Business", "Sci/Tech"]

# Streamlit UI
st.title(" BERT News Classifier")
st.write("Enter a news headline or sentence, and BERT will predict its category.")

# Text input
user_input = st.text_area(" Enter News Text", height=150)

if st.button("Predict"):
    if not user_input.strip():
        st.warning(" Please enter some text.")
    else:
        # Tokenize input
        inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True)

        # Get prediction
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            predicted_label = torch.argmax(probs, dim=1).item()
            confidence = probs[0][predicted_label].item()

        # Show result
        st.success(f" **Predicted Category**: {labels[predicted_label]}")
        st.write(f" **Confidence**: {confidence:.2%}")
