import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os
import tempfile
import base64
import json
from elevenlabs.client import ElevenLabs

from rag_module import (
    extract_text,
    split_text,
    create_vector_store,
    answer_query_with_context,
    advanced_retrieval,
    verify_answer,
    evaluate_answer,
    evaluate_multimodal_response
)

from MultimodInput import get_user_query
from diagramgen import generate_diagram_streamlit
from agent_controller import detect_intent

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
eleven_api_key = os.getenv("ELEVEN_API_KEY")

if not groq_api_key:
    try:
        groq_api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass

if not openai_api_key:
    try:
        openai_api_key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass

if not eleven_api_key:
    try:
        eleven_api_key = st.secrets["ELEVEN_API_KEY"]
    except Exception:
        pass

st.set_page_config(page_title="ClarifAI", layout="centered")

st.title("🤖 ClarifAI - AI Powered Learning Assistant")
st.caption("Your AI Guru, Guiding You from Doubt to Wisdom.")

with st.sidebar:
    model = st.selectbox(
        "Choose Model",
        [
            "llama-3.3-70b-versatile",
            "llama3-70b-8192",
            "openai/gpt-oss-120b",
        ],
        index=0,
    )

    input_mode = st.radio("Input Type", ["Text", "Image", "Voice"])
    speak_response = st.checkbox("🔊 Enable AI Voice Output", value=False)

    generate_diagram_flag = st.checkbox("🧠 Generate Concept Diagram")
    generate_image_flag = st.checkbox("🎨 Generate Image")

    st.subheader("📄 Document Q&A")

    uploaded_file = st.file_uploader(
        "Upload PDF, DOCX, or TXT",
        type=["pdf", "docx", "txt"]
    )

    use_doc_context = st.checkbox("Use document context")

    level = st.selectbox(
        "🎓 Learning Level",
        ["Beginner", "Intermediate", "Advanced"]
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
    st.session_state.doc_chunks = []

if "quiz_data" not in st.session_state:
    st.session_state.quiz_data = None

user_query = get_user_query(input_mode)
intent = detect_intent(user_query) if user_query else None

extracted_text = None
if input_mode in ["Image", "Voice"]:
    extracted_text = user_query

# =====================================================
# 🎯 QUIZ GENERATION
# =====================================================
if user_query and any(word in user_query.lower() for word in ["quiz", "test", "mcq"]):

    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=groq_api_key,
    )

    if use_doc_context and st.session_state.vector_store:
        context = "\n".join(st.session_state.doc_chunks[:10])
        topic_prompt = f"Based on this content generate quiz:\n{context}"
    else:
        topic_prompt = f"Generate quiz on: {user_query}"

    prompt = f"""
Create a 5-question multiple choice quiz.

Return ONLY valid JSON in this format:

[
  {{
    "question": "...",
    "options": ["A", "B", "C", "D"],
    "answer": "correct option text"
  }}
]

Topic:
{topic_prompt}
"""

    res = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )

    try:
        st.session_state.quiz_data = json.loads(res.choices[0].message.content)
    except Exception:
        st.error("Quiz generation failed")
        st.stop()

# =====================================================
# 🏆 QUIZ UI
# =====================================================
if st.session_state.quiz_data:

    st.header("📝 Quiz Time!")

    user_answers = []

    for i, q in enumerate(st.session_state.quiz_data):
        st.markdown(
            f"<div style='background:#1f2937;padding:15px;border-radius:10px;margin-bottom:12px'><b>Q{i+1}. {q['question']}</b></div>",
            unsafe_allow_html=True
        )

        ans = st.radio("Select answer:", q["options"], key=f"quiz_{i}")
        user_answers.append(ans)

    if st.button("Submit Quiz 🚀"):

        score = sum(
            ua == q["answer"]
            for ua, q in zip(user_answers, st.session_state.quiz_data)
        )

        st.success(f"🎯 Score: {score} / {len(user_answers)}")

        st.subheader("✅ Correct Answers")
        for i, q in enumerate(st.session_state.quiz_data):
            st.write(f"Q{i+1}: {q['answer']}")

    st.stop()

# =====================================================
# IMAGE GENERATION
# =====================================================
if user_query and (generate_image_flag or intent == "image"):
    if openai_api_key:
        image_client = OpenAI(api_key=openai_api_key)
        img_response = image_client.images.generate(
            model="gpt-image-1",
            prompt=user_query,
            size="1024x1024",
        )
        st.image(img_response.data[0].url)
    st.stop()

# =====================================================
# DIAGRAM
# =====================================================
if groq_api_key and user_query and (generate_diagram_flag or intent == "diagram"):
    diagram_path = generate_diagram_streamlit(user_query)

    if diagram_path:
        with open(diagram_path, "rb") as img_file:
            img_b64 = base64.b64encode(img_file.read()).decode()

        st.markdown(
            f'<img src="data:image/png;base64,{img_b64}" style="max-width:100%"/>',
            unsafe_allow_html=True,
        )
    st.stop()

# =====================================================
# DOCUMENT PROCESS
# =====================================================
if uploaded_file and "processed_file_name" not in st.session_state:
    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    raw_text = extract_text(file_path)
    chunks = split_text(raw_text)
    vector_store = create_vector_store(chunks)

    st.session_state.vector_store = vector_store
    st.session_state.doc_chunks = chunks
    st.session_state.processed_file_name = uploaded_file.name

# =====================================================
# NORMAL CHAT
# =====================================================
if groq_api_key and user_query:

    st.session_state.messages.append(
        {"role": "user", "content": user_query}
    )

    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=groq_api_key,
    )

    if use_doc_context and st.session_state.vector_store:

        context_chunks = advanced_retrieval(
            user_query,
            st.session_state.vector_store,
            st.session_state.doc_chunks
        )

        st.subheader("🔎 Retrieved Context")
        for i, chunk in enumerate(context_chunks, 1):
            st.markdown(f"**Chunk {i}:** {chunk[:300]}...")

        adapted_query = f"{level} explanation: {user_query}"
        reply = answer_query_with_context(adapted_query, context_chunks)

        verdict = verify_answer(reply, context_chunks)
        st.info(f"🧠 Self-check: {verdict}")

        metrics = evaluate_answer(reply, context_chunks)

        st.subheader("📊 RAG Evaluation")
        st.write(metrics)

    else:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": user_query}],
        )
        reply = response.choices[0].message.content

    st.session_state.messages.append(
        {"role": "assistant", "content": reply}
    )

    acc_metrics = evaluate_multimodal_response(
        query=user_query,
        answer=reply,
        context_chunks=st.session_state.doc_chunks if use_doc_context else None,
        input_mode=input_mode,
        extracted_text=extracted_text,
        intent=intent
    )

    st.subheader("📈 Overall Response Accuracy")
    st.write(acc_metrics)

    if speak_response and eleven_api_key:
        eleven_client = ElevenLabs(api_key=eleven_api_key)
        audio_generator = eleven_client.text_to_speech.convert(
            voice_id="21m00Tcm4TlvDq8ikWAM",
            text=reply,
            model_id="eleven_flash_v2",
            output_format="mp3_44100_128",
        )
        audio_bytes = b"".join(audio_generator)
        st.audio(audio_bytes)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if not groq_api_key:
    st.info("🔑 Please add your Groq API key.")
