import streamlit as st
import os
from PIL import Image
import easyocr
import numpy as np
from openai import OpenAI

# ---------------------------------
# OCR (cached)
# ---------------------------------
@st.cache_resource
def load_ocr_reader():
    return easyocr.Reader(["en"], gpu=False)

ocr_reader = load_ocr_reader()

# ---------------------------------
# Groq client (cached)
# ---------------------------------
@st.cache_resource
def get_groq_client():
    return OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.getenv("GROQ_API_KEY"),
    )

# ---------------------------------
# Main multimodal input handler
# ---------------------------------
def get_user_query(input_mode, record_button_pressed=False):

    # ---------- TEXT ----------
    if input_mode == "Text":
        return st.chat_input("Type your message here...")

    # ---------- IMAGE ----------
    if input_mode == "Image":
        st.markdown("📤 Upload an image:")
        uploaded_image = st.file_uploader(
            "Upload image", type=["jpg", "png", "jpeg"]
        )

        if uploaded_image:
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_container_width=True)

            with st.spinner("🔍 Extracting text..."):
                img_np = np.array(image)
                results = ocr_reader.readtext(img_np)

                if results:
                    text = " ".join([t for _, t, _ in results])
                    st.success("✅ Text extracted")
                    return text.strip()
                else:
                    st.warning("⚠️ No readable text found")

        return None

    # ---------- VOICE ----------
    if input_mode == "Voice":

        # Initialize session state
        if "voice_text" not in st.session_state:
            st.session_state.voice_text = None

        st.markdown("🎙️ Speak and submit your voice")

        audio = st.audio_input(
            label="Voice Recorder",
            label_visibility="collapsed",
        )


        if audio and st.session_state.voice_text is None:
            with st.spinner("🧠 Transcribing speech..."):
                try:
                    client = get_groq_client()
                    audio_bytes = audio.read()

                    transcript = client.audio.transcriptions.create(
                        file=("audio.wav", audio_bytes),
                        model="whisper-large-v3",
                        response_format="text",
                    )

                    st.session_state.voice_text = transcript.strip()
                    st.success("✅ Transcription complete")

                except Exception as e:
                    st.error(f"❌ Speech-to-text failed: {e}")

        return st.session_state.voice_text
