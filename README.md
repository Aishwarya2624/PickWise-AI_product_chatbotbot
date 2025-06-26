# PickWise-AI_product_chatbotbot
This project is an **offline smart chatbot** that answers user questions about mobile phones using local machine learning models — with **no internet, no API calls, and no external costs**.

---

## 🔧 Features

- 💬 Chatbot-style interface using **Streamlit**
- 🧠 Semantic product matching using **TF-IDF**
- 🤖 Local response generation using **FLAN-T5**
- 🗂️ Product data from a simple CSV file
- 🔍 Sidebar search filter for quick product discovery
- ✅ Works entirely **offline**

---


## 🛠️ Tech Stack

- Python
- Streamlit
- Transformers (flan-t5-base)
- Scikit-learn (TF-IDF similarity)
- Pandas

---

## 📦 How to Run

1.clone this repo:
git clone https://github.com/YOUR_USERNAME/ai-product-chatbot-offline.git
cd ai-product-chatbot-offline


2.Install dependencies:
pip install -r requirements.txt

3. Download the FLAN-T5 model:
   from transformers import pipeline
pipeline("text2text-generation", model="google/flan-t5-base")

4.Run the App:
streamlit run app.py

📌 Why I Built This
To demonstrate how local LLMs like flan-t5-base can be combined with smart retrieval (TF-IDF) to build a fully functional chatbot — without relying on OpenAI or paid APIs. It's ideal for edge use-cases like offline retail, kiosks, or privacy-sensitive environments.


