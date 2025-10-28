from google import genai
from google.genai import types
from google.genai.errors import APIError
import streamlit as st
import numpy as np
import pandas as pd

def embed(client, texts):
    embeddings = [
        np.array(e.values) for e in client.models.embed_content(
            model='text-embedding-004',
            contents= texts,
            config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
        ).embeddings
    ]
    return embeddings

def cosine_similarity(embeddings):
    embeddings = np.array(embeddings)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings/norms
    similarity_matrix = np.dot(normalized, normalized.T)
    return similarity_matrix

def gemini_qa(client):
    st.header("💬 Question-Answering Chatbot using Gemini")

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    if "client" not in st.session_state:
        st.session_state["client"] = client
    if "chat" not in st.session_state:
        st.session_state["chat"] = st.session_state["client"].chats.create(model=MODEL_NAME)

    chat_container = st.container()

    with chat_container:
        for msg in st.session_state["messages"]:
            st.chat_message(msg["role"]).write(msg["content"])

    prompt = st.chat_input("Ask me anything...")

    if prompt:
        st.session_state["messages"].append({"role": "user", "content": prompt})
        
        with chat_container:
            st.chat_message("user").write(prompt)

        try:
            response = st.session_state["chat"].send_message(prompt)
            msg = response.text
        except APIError as e:
            msg = f"[API ERROR] {e}"
        except Exception as e:
            msg = f"[ERROR] {e}"

        st.session_state["messages"].append({"role": "assistant", "content": msg})
        with chat_container:
            st.chat_message("assistant").write(msg)

def gemini_embeddings(client):
    st.header("🔍 Embeddings for Text Similarity using Gemini")

    num = int(st.number_input("Input Number of Sentences:", min_value=1, value=1))

    text_inputs = []
    for i in range(num):
        text_inputs.append(st.text_input(f"Enter sentence {i+1}", key=f"sentence_{i+1}"))

    if st.button("Save sentences"):
        sentences = [s for s in text_inputs if s and s.strip()]
        st.session_state["sentences"] = sentences
        st.success(f"Saved {len(sentences)} sentence(s).")

    if "sentences" in st.session_state:
        st.write("Stored sentences:", st.session_state["sentences"])

    cosine_similarity_matrix = cosine_similarity(embed(client=client, texts=text_inputs))

    labels = [s if s and s.strip() else f"Sentence {i+1}" for i, s in enumerate(text_inputs)]
    df = pd.DataFrame(cosine_similarity_matrix, columns=labels, index=labels)

    if "sentences" in st.session_state and st.session_state["sentences"]:
        st.dataframe(df, use_container_width=True)

if __name__ == "__main__":
    st.title("🤖 Gemini SDK Playground")

    with st.sidebar:
        API_KEY = st.text_input("Enter API Key:")
        st.header("About this project")
        st.write(
            "Gemini SDK Playground is a small Streamlit app that demonstrates two features of the Google Gemini SDK:\n\n"
            "- Chat-based question-answering using a Gemini chat model.\n"
            "- Generating text embeddings and computing cosine similarity between sentences.\n\n"
            "Usage:\n"
            "1. Use the 'Gemini SDK' tab to chat with the model.\n"
            "2. Use the 'Embeddings' tab to enter sentences and compute similarities.\n\n"
            "Security note: Please add your own Gemini API key in the sidebar."
            "To obtain an API key and learn how to use Google AI Studio, see https://studio.google.ai/ and the Generative AI docs at https://developers.generativeai.google/."
        )
        st.caption("Built with google-genai and Streamlit")

    MODEL_NAME = "gemini-2.5-flash"
    client = genai.Client(api_key=API_KEY)

    tab1, tab2 = st.tabs(["QA Chat", "Embeddings"])
    with tab1:
        gemini_qa(client)
    with tab2:
        gemini_embeddings(client)
    
