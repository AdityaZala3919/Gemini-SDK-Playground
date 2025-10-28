# 🤖 Gemini SDK Playground

An interactive **Streamlit app** built with the **Google Gemini SDK**, demonstrating:

- 💬 **Question-Answering Chatbot** using the Gemini chat model  
- 🔍 **Text Embeddings and Semantic Similarity** using Gemini embeddings  

---

## 🚀 Features

### 💬 1. Gemini Chatbot
- Uses the `gemini-2.5-flash` model for real-time Q&A.  
- Maintains conversation history using Streamlit session state.  
- Clean chat interface built with `st.chat_message()`.  
- Handles errors gracefully (API or runtime).

### 🔍 2. Text Embeddings and Similarity
- Generates embeddings using the `text-embedding-004` model.  
- Computes **cosine similarity** between multiple sentences.  
- Displays results in a structured **pandas DataFrame** for visualization.  

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

### 1. Clone the repository
```bash
git clone https://github.com/AdityaZala3919/Gemini-SDK-Playground.git
cd Gemini-SDK-Playground
````

### 2. (Optional) Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate    # Mac/Linux
venv\Scripts\activate       # Windows
```

### 3. Install dependencies

```bash
pip install streamlit google-genai numpy pandas
```

---

## 🔑 Setup Gemini API Key

1. Go to [Google AI Studio](https://aistudio.google.com/)
2. Get your **Gemini API key**
3. Paste it into the sidebar input in the app

---

## ▶️ Run the App

```bash
streamlit run main.py
```

After running, open the provided URL (usually `http://localhost:8501`).

---

## 🧠 How It Works

### 🔹 Chatbot

* Initializes a Gemini chat session:

  ```python
  chat = client.chats.create(model="gemini-2.5-flash")
  ```
* Handles user prompts and responses dynamically.

### 🔹 Embeddings

* Generates embeddings using:

  ```python
  client.models.embed_content(model="text-embedding-004", contents=texts)
  ```
* Computes cosine similarity using NumPy:

  ```python
  np.dot(normalized_vectors, normalized_vectors.T)
  ```

---

## 🧾 Example Use Cases

* Build and test **Gemini-based conversational apps**
* Compare **semantic similarity** between sentences
* Prototype **AI tools** that use embeddings (RAG, clustering, etc.)
* Learn **Google Gemini SDK** integration in Streamlit

---

## 🧰 Project Structure

```
gemini-sdk-playground/
│
├── main.py          # Main Streamlit app file
└── README.md        # Project documentation
```

---

## 💡 Future Enhancements

* Add file upload support for Q&A
* Visualize embeddings using t-SNE or PCA
* Add chat export (Markdown/PDF)
* Support multiple Gemini model selections

---

## 🧑‍💻 Author

**Adityasinh Zala** <br>
AI/ML Engineer | Tech Explorer | Curious Learner   <br>
[GitHub](https://github.com/AdityaZala3919) • [LinkedIn](https://www.linkedin.com/in/adityasinh-zala-1bbb42258/)

---

⭐ *If you found this helpful, don’t forget to give it a star on GitHub!*
🌐 *Built with ❤️ using Google Gemini SDK and Streamlit.*
