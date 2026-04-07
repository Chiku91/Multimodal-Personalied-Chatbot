import streamlit as st
import os
from PIL import Image
import easyocr
import numpy as np
from openai import OpenAI
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tempfile

# ✅ GLOBAL SAFE INITIALIZATION
if "voice_text" not in st.session_state:
    st.session_state.voice_text = None

if "last_audio_id" not in st.session_state:
    st.session_state.last_audio_id = None

@st.cache_resource
def load_ocr_reader():
    return easyocr.Reader(["en"], gpu=False)

ocr_reader = load_ocr_reader()

@st.cache_resource
def get_groq_client():
    return OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.getenv("GROQ_API_KEY"),
    )

# 🔥 MEL SPECTROGRAM
def generate_mel_spectrogram(audio_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        temp_path = tmp.name

    y, sr = librosa.load(temp_path, sr=None)

    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)

    fig, ax = plt.subplots()
    img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=ax)
    fig.colorbar(img, ax=ax)
    ax.set_title("🎧 Mel Spectrogram")

    return fig

# ---------------------------------
# MAIN FUNCTION
# ---------------------------------
def get_user_query(input_mode, record_button_pressed=False):

    if input_mode == "Text":
        return st.chat_input("Type your message here...")

    # ================= IMAGE =================
    if input_mode == "Image":

        st.markdown("📤 Upload an image:")

        uploaded_image = st.file_uploader(
            "Upload image", type=["jpg", "png", "jpeg"]
        )

        extracted_text = ""

        if uploaded_image:
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_container_width=True)

            with st.spinner("🔍 Extracting text from image..."):
                img_np = np.array(image)
                results = ocr_reader.readtext(img_np)

                if results:
                    extracted_text = " ".join([t for _, t, _ in results])
                    st.success("✅ Text extracted")
                    st.text_area("📄 Extracted Text", extracted_text, height=150)
                else:
                    st.warning("⚠️ No readable text found")

        user_prompt = st.text_input("💬 Ask something about the image:")

        if st.button("Submit Image Query 🚀"):
            if user_prompt:
                return f"{user_prompt}\n\n[Image Content]: {extracted_text}" if extracted_text else user_prompt

        return None

    # ================= VOICE =================
    if input_mode == "Voice":

        st.markdown("🎙️ Speak and submit your voice")

        audio = st.audio_input(
            label="Voice Recorder",
            label_visibility="collapsed",
        )

        if audio:

            # 🔥 Detect new audio (important fix)
            audio_id = hash(audio.getvalue())

            if st.session_state.last_audio_id != audio_id:
                st.session_state.voice_text = None
                st.session_state.last_audio_id = audio_id

            if st.session_state.voice_text is None:
                with st.spinner("🧠 Processing voice..."):
                    try:
                        client = get_groq_client()
                        audio_bytes = audio.read()

                        # 🎧 MEL SPECTROGRAM
                        st.subheader("🎧 Mel Spectrogram")
                        fig = generate_mel_spectrogram(audio_bytes)
                        st.pyplot(fig)

                        # 🧠 TRANSCRIPTION
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