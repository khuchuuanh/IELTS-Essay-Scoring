import streamlit as st
import requests
from typing import Dict

class TextData:
    def __init__(self, text: str):
        self.text = text

    def to_dict(self) -> Dict:
        return {"text": self.text}

st.title("IELTS Scoring App")

question = st.text_area("Enter the question", height=100)
essay = st.text_area("Enter the essay", height=550)

if st.button("Predict"):
    input_data = {"question": TextData(question).to_dict(), "essay": TextData(essay).to_dict()}
    response = requests.post("http://localhost:8000/predict", json=input_data)

    if response.status_code == 200:
        prediction = response.json()
        st.write("Predicted Coherence:", prediction["predicted_coherence"])
        st.write("Predicted Lexical:", prediction["predicted_lexical"])
        st.write("Predicted Grammar:", prediction["predicted_grammar"])
        st.write("Predicted Task:", prediction["predicted_task"])

    else:
        st.write("Error:", response.text)
