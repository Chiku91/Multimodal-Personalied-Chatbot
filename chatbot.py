import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os
import tempfile
import base64
from elevenlabs.client import ElevenLabs
from rag_module import (
    extract_text,
    split_text,
    create_vector_store,
    advanced_retrieval,
    answer_query_with_context,
)
from MultimodInput import get_user_query
from diagramgen import generate_diagram_streamlit

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

    st.divider()

    generate_diagram_flag = st.checkbox("🧠 Generate Concept Diagram")
    generate_image_flag = st.checkbox("🎨 Generate Image")

    st.divider()

    st.subheader("📄 Document Q&A")
    uploaded_file = st.file_uploader(
        "Upload PDF, DOCX, or TXT", type=["pdf", "docx", "txt"]
    )
    use_doc_context = st.checkbox("Use document context")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
    st.session_state.doc_chunks = []
    st.session_state.raw_text = ""

user_query = get_user_query(input_mode)

if user_query and generate_image_flag:
    if openai_api_key:
        try:
            image_client = OpenAI(api_key=openai_api_key)
            img_response = image_client.images.generate(
                model="gpt-image-1",
                prompt=user_query,
                size="1024x1024",
            )
            image_url = img_response.data[0].url
            st.image(image_url, use_container_width=True)
            st.markdown(f"[Download Image]({image_url})")
        except Exception as e:
            st.error(f"Image generation failed: {e}")
    st.stop()

if groq_api_key and user_query and generate_diagram_flag:
    diagram_path = generate_diagram_streamlit(user_query)

    if diagram_path:
        with open(diagram_path, "rb") as img_file:
            img_bytes = img_file.read()
            img_b64 = base64.b64encode(img_bytes).decode()

        st.markdown(
            f"""
            <img src="data:image/png;base64,{img_b64}" style="max-width:100%;">
            """,
            unsafe_allow_html=True,
        )
    st.stop()

if uploaded_file:
    if (
        "processed_file_name" in st.session_state
        and st.session_state.processed_file_name != uploaded_file.name
    ):
        for key in [
            "vector_store",
            "doc_chunks",
            "raw_text",
            "processed_file_name",
        ]:
            st.session_state.pop(key, None)

if uploaded_file and "processed_file_name" not in st.session_state:
    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    raw_text = extract_text(file_path)
    chunks = split_text(raw_text)
    vector_store = create_vector_store(chunks)

    st.session_state.vector_store = vector_store
    st.session_state.doc_chunks = chunks
    st.session_state.raw_text = raw_text
    st.session_state.processed_file_name = uploaded_file.name

if groq_api_key and user_query:
    st.session_state.messages.append(
        {"role": "user", "content": user_query}
    )

    try:
        client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=groq_api_key,
        )

        if use_doc_context and st.session_state.vector_store:
            context_chunks = advanced_retrieval(
                user_query,
                st.session_state.vector_store,
                st.session_state.doc_chunks,
            )
            reply = answer_query_with_context(user_query, context_chunks)
        else:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": user_query}],
                temperature=0.7,
            )
            reply = response.choices[0].message.content

        st.session_state.messages.append(
            {"role": "assistant", "content": reply}
        )

        if speak_response and eleven_api_key and reply.strip():
            try:
                eleven_client = ElevenLabs(api_key=eleven_api_key)
                audio = eleven_client.text_to_speech.convert(
                    voice_id="21m00Tcm4TlvDq8ikWAM",
                    text=reply,
                    model_id="eleven_flash_v2",
                    output_format="mp3_44100_128",
                )
                audio_bytes = b"".join(audio)
                st.audio(audio_bytes, format="audio/mp3")
            except Exception:
                pass

    except Exception as e:
        st.error(f"Error: {e}")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if not groq_api_key:
    st.info("Please add your Groq API key")
