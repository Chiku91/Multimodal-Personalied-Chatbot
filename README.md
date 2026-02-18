# 🤖 **ClarifAi - AI-Powered Learning Assistant**<br>

"Your AI Guru, Guiding You from Doubt to Wisdom. The Light of Knowledge in a Digital Avatar."

Mentora is a Streamlit-based multifunctional AI chatbot tailored for learners, educators, and curious minds. It seamlessly integrates multimodal input, voice response, document summarization, diagram generation, and RAG-based document Q&A, all powered by modern LLMs like Groq and Mistral.

## 🚀 **Features**

### 🧠 Chatbot Powered by Groq LLMs</br>

- Dynamic, human-like conversations.</br>
- Option to choose models like `mistral-saba-24b` and `llama3-70b-8192`.

### 🎙️ Multimodal Input (MultimodInput.py)</br>

- Text: Regular chat input.</br>
- Voice: `Whisper model` transcribes spoken queries.</br>
- Image: `OCR-based` extraction using `Tesseract` for image-to-text questions.</br>

### 📊 Concept Diagram Generator (diagramgen.py)</br>

- Generate concept visualizations based on query context using LLM-driven classification and rendering libraries like `Graphviz` and `Matplotlib`.</br>
- Dynamically rendered inside the Streamlit app with download options.</br>

### 📚 RAG-based Document Q&A (rag_module.py)</br>

- Upload PDFs, DOCX, or TXT files.</br>
- Query content semantically using `FAISS`, `Sentence Transformers`, and `Groq`.</br>
- Context-aware response generation via `Retrieval-Augmented Generation`.</br>
- Summarize uploaded documents with RAG-powered context understanding.</br>

### 🔈 Voice Output</br>

- Text responses can be read aloud using `ElevenLabs API` for immersive interaction.</br>

## 🛠️ Setup Instructions

1. Clone the Repository
<pre>
git clone https://github.com/developer-Shaurya/ai_assistant.git
cd ai_assistant
</pre>
2. Create a Virtual Environment
<pre>
  python -m venv venv
  source venv/bin/activate      # On Windows: venv\Scripts\activate
</pre>

3. Install Dependencies
<pre>
  pip install -r requirements.txt
</pre>

4. Setup Environment Variables
   Create a `.env` file:

<pre>
  GROQ_API_KEY=your_groq_api_key
  ELEVEN_API_KEY=your_elevenlabs_api_key
</pre>

Running the App

<pre>
  streamlit run chatbot.py
</pre>
