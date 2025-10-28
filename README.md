# 🤖 Gemini SDK Playground

A **Streamlit-based interactive playground** built using the **Google Gemini SDK**, demonstrating:
- 💬 **Chat-based Question Answering** using the Gemini chat model.  
- 🔍 **Text Embeddings and Similarity Search** using Gemini embeddings.

---

## 🚀 Features

### 1. 💬 Question-Answering Chatbot
- Uses the `gemini-2.5-flash` model.
- Provides a conversational chat interface with memory.
- Handles API errors gracefully and maintains chat history.
- Built with Streamlit’s `st.chat_message()` for a modern chat UI.

### 2. 🔍 Text Embedding & Similarity
- Generates **text embeddings** using `text-embedding-004`.
- Computes **cosine similarity** between multiple sentences.
- Displays similarity as an interactive **pandas DataFrame**.
- Helps visualize semantic relationships between user-provided texts.

---

## 🧩 Tech Stack

| Component | Description |
|------------|-------------|
| **Language** | Python 3.9+ |
| **Framework** | [Streamlit](https://streamlit.io/) |
| **AI SDK** | [Google Gemini SDK (`google-genai`)](https://googleapis.github.io/python-genai/) |
| **Libraries** | `numpy`, `pandas` |

---

## 📦 Installation

### 1. Clone this repository
```bash
git clone https://github.com/<your-username>/gemini-sdk-playground.git
cd gemini-sdk-playground
````

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate   # (Linux/Mac)
venv\Scripts\activate      # (Windows)
```

### 3. Install dependencies

```bash
pip install streamlit google-genai numpy pandas
```

---

## 🔑 Setting Up API Access

1. Visit [Google AI Studio](https://aistudio.google.com/) to get your **Gemini API key**.
2. Copy your key and paste it into the sidebar input in the Streamlit app.

---

## ▶️ Running the App

```bash
streamlit run app.py
```

Then open the provided local URL in your browser — usually:

```
http://localhost:8501
```

---

## 🧠 App Structure

```
gemini-sdk-playground/
│
├── app.py                # Main Streamlit app
│
├── functions/
│   ├── gemini_chat.py    # Chat functionality (Q&A)
│   ├── gemini_embed.py   # Embedding + cosine similarity
│   └── config.py         # Client and model setup
│
├── requirements.txt
└── README.md
```

*(If all code is inside one file, structure can be simplified to just `app.py`.)*

---

## 💡 How It Works

### 🔹 Chatbot

* Creates a chat session with Gemini using:

  ```python
  client.chats.create(model="gemini-2.5-flash")
  ```
* Maintains conversation context via `st.session_state`.

### 🔹 Embeddings

* Uses:

  ```python
  client.models.embed_content(model="text-embedding-004", contents=texts)
  ```
* Computes **cosine similarity**:

  ```python
  sim = np.dot(normalized_vectors, normalized_vectors.T)
  ```

---

## 🧾 Example Use Cases

* Build **interactive chatbots** with Gemini API.
* Test **semantic similarity** of phrases or sentences.
* Learn how to integrate **Google GenAI SDK** in Python apps.
* Prototype **RAG systems** or **document search tools** using embeddings.

---

## 🧰 Future Enhancements

* Add **file upload** support for text-based Q&A.
* Visualize **embedding clusters** with t-SNE or PCA.
* Include **chat history download/export**.
* Support **multiple Gemini models** (e.g., `gemini-1.5-pro`).

---

## 🧑‍💻 Author

**Aditya Zala**
AI/ML Engineer | Developer | Research Enthusiast
[LinkedIn](https://www.linkedin.com/) • [GitHub](https://github.com/<your-username>)

---

## 📜 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

### ⭐ If you find this helpful, give it a star on GitHub!

```
🌐 Built with ❤️ using Google Gemini SDK and Streamlit.
```

Would you like me to make the **`requirements.txt`** file too (with pinned versions for each dependency)?
