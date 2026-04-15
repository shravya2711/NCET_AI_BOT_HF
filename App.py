import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

@st.cache_resource
def load_summarizer():
    model_name = "sshleifer/distilbart-cnn-12-6"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    return pipeline(
        task="summarization",   
        model=model,
        tokenizer=tokenizer
    )

summarizer = load_summarizer()

st.title("AI Text Summarizer")

text = st.text_area("Enter summarizing text:")

if st.button("Summarize"):
    if text.strip():
        result = summarizer(text, max_length=130, min_length=30)
        st.write(result[0]["summary_text"])
    else:
        st.warning("Please enter some text to summarize.")
