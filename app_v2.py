import streamlit as st
from groq import Groq
from gtts import gTTS
from io import BytesIO
from PIL import Image
import requests
import base64

st.title("PCM Problem Solver with Groq API and gTTS")

# 1. Subject selection
if 'subject' not in st.session_state:
    st.session_state.subject = None
if 'groq_api_key' not in st.session_state:
    st.session_state.groq_api_key = None
if 'problem_text' not in st.session_state:
    st.session_state.problem_text = None
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'conversation' not in st.session_state:
    st.session_state.conversation = []
if 'solved' not in st.session_state:
    st.session_state.solved = False
if 'followup_key' not in st.session_state:
    st.session_state.followup_key = 0

# Step 1: User input for subject, API key, problem (text or image)
if not st.session_state.solved:
    st.session_state.subject = st.selectbox("Solve problem for:", ["Physics", "Chemistry", "Maths"])
    st.session_state.groq_api_key = st.text_input("Enter your Groq API key:", type="password")
    st.session_state.problem_text = st.text_area("Enter your problem (optional):")
    st.session_state.uploaded_image = st.file_uploader("Upload an image or take a picture (optional)", type=["jpg", "jpeg", "png"])

    def ocr_space_file_upload(image_file, api_key='K88884750088957'):
        payload = {
            'isOverlayRequired': False,
            'apikey': api_key,
            'language': 'eng',
        }
        files = {'file': image_file}
        r = requests.post('https://api.ocr.space/parse/image', files=files, data=payload)
        result = r.json()
        try:
            return result['ParsedResults'][0]['ParsedText']
        except Exception:
            return ""

    if st.button("Solve Problem"):
        if not st.session_state.groq_api_key:
            st.warning("Please enter your Groq API key.")
        else:
            input_to_solve = st.session_state.problem_text.strip() if st.session_state.problem_text else ""
            if st.session_state.uploaded_image is not None:
                extracted_text = ocr_space_file_upload(st.session_state.uploaded_image)
                if extracted_text:
                    if input_to_solve:
                        input_to_solve += "\n" + extracted_text
                    else:
                        input_to_solve = extracted_text
            if input_to_solve:
                # Prepare conversation
                st.session_state.conversation = [
                    {"role": "user", "content": input_to_solve}
                ]
                # Call Groq
                client = Groq(api_key=st.session_state.groq_api_key)
                expert_instruction = (
                    f"You are a {st.session_state.subject} expert.\n"
                        "You will be given a question and you need to solve it step by step.\n"
                        "Answer in a mix of English and Hindi as if you are an Indian teacher explaining to a student.\n"
                        "try to give your answer in 500 tokens only"
                )
                groq_messages = [
                    {"role": "system", "content": expert_instruction},
                    {"role": "user", "content": input_to_solve}
                ]
                completion = client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=groq_messages,
                    temperature=1,
                    max_completion_tokens=512,
                    top_p=1,
                    stream=True,
                    stop=None,
                )
                result = ""
                for chunk in completion:
                    result += chunk.choices[0].delta.content or ""
                st.session_state.conversation.append({"role": "assistant", "content": result})
                st.session_state.solved = True
            else:
                st.warning("Please provide a problem in text or upload an image.")

# Step 2: Show solution and allow only text follow-up questions
if st.session_state.solved:
    st.subheader("Conversation:")
    for i, msg in enumerate(st.session_state.conversation):
        if msg["role"] == "user":
            st.markdown(f"**You:** {msg['content']}")
        elif msg["role"] == "assistant":
            st.markdown(f"**PCM Solver:** {msg['content']}")
            try:
                tts = gTTS(msg["content"])
                audio_fp = BytesIO()
                tts.write_to_fp(audio_fp)
                audio_fp.seek(0)
                st.audio(audio_fp, format='audio/mp3')
            except Exception as e:
                st.warning(f"Audio unavailable for this message: {e}")

    # Only text input for follow-up questions
    followup = st.text_input("Ask a follow-up question:", key=f"followup_q_{st.session_state.followup_key}")
    if st.button("Send Follow-up"):
        if followup.strip():
            st.session_state.conversation.append({"role": "user", "content": followup})
            # Prepare conversation for Groq
            groq_messages = [
                {"role": "system", "content": f"You are a {st.session_state.subject} expert.\nYou will be given a question and you need to solve it step by step.\nAnswer in a mix of English and Hindi as if you are an Indian teacher explaining to a student.\n"}
            ]
            for msg in st.session_state.conversation:
                groq_messages.append(msg)
            client = Groq(api_key=st.session_state.groq_api_key)
            completion = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=groq_messages,
                temperature=1,
                max_completion_tokens=512,
                top_p=1,
                stream=True,
                stop=None,
            )
            result = ""
            for chunk in completion:
                result += chunk.choices[0].delta.content or ""
            st.session_state.conversation.append({"role": "assistant", "content": result})
            st.session_state.followup_key += 1  # This will reset the input box 
