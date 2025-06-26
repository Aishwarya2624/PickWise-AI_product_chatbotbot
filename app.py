import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

# Load product data
df = pd.read_csv("product_data.csv")

# TF-IDF vectorization
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['description'])

# Load the local language model
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")

# Streamlit layout
st.set_page_config(page_title="Offline Chatbot with Search", layout="wide")
st.title("🤖 PickWise")
st.caption("Powered by TF-IDF + FLAN-T5")

with st.sidebar:
    st.header("🔍 Filter Products")
    keyword = st.text_input("Search by keyword")
    if keyword:
        filtered_df = df[df['description'].str.contains(keyword, case=False) | df['title'].str.contains(keyword, case=False)]
        st.write("Matching Products:")
        for _, row in filtered_df.iterrows():
            st.markdown(f"- **{row['title']}**: {row['description']}")
    else:
        st.write("Type a keyword to filter phones.")

# Question input
st.markdown("### 💬 Ask a product-related question")
user_input = st.text_input("Your question:")

if user_input:
    # Find best context using TF-IDF
    query_vec = vectorizer.transform([user_input])
    scores = cosine_similarity(query_vec, tfidf_matrix)
    top_idx = scores[0].argmax()
    context = df.iloc[top_idx]['description']
    title = df.iloc[top_idx]['title']

    # Generate response using LLM
    prompt = f"Context: {context} \nQuestion: {user_input}"
    result = qa_pipeline(prompt, max_length=100)[0]['generated_text']

    # Display response
    st.markdown(f"🧑‍💬 **You:** {user_input}")
    st.markdown(f"🤖 **Bot: Product:** **{title}**\n\n{result}")
