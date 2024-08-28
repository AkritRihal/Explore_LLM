import requests 
import streamlit as st
def get_openai_response(input_text):
    response = requests.post("http://localhost:8000/essay/invoke",
                             json = {input : {'topic: input_text'}}
                             )
    return response.json()['output']

st.title("LAngchain Demo with LLAMA API")
input_text = st.text_input("write an poem on")


if input_text:
    st.write(get_openai_response)