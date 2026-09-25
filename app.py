import streamlit as st
from openai import OpenAI

st.title("Retrieval-Augmented Generation for Erythromelalgia")

client_openai = OpenAI(
    api_key=st.secrets["OPENAI_API_KEY"]
)

client_deepseek = OpenAI(
    api_key=st.secrets["BASETEN_API_KEY"],
    base_url="https://inference.baseten.co/v1"
)

vector_store_id = st.secrets["OPENAI_VECTOR_STORE_ID"]

try:
    vector_store = client_openai.vector_stores.retrieve(
        vector_store_id=vector_store_id
    )

    st.success("Vector Store connected successfully.")

except Exception as e:
    st.error("Unable to connect to the Vector Store.")
    st.code(str(e))
