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

st.subheader("System check")

# OpenAI
try:
    client_openai.models.list()
    st.success("OpenAI ✓")
except Exception as e:
    st.error("OpenAI ✗")
    st.code(str(e))


# Vector Store
try:
    vector_store = client_openai.vector_stores.retrieve(
        vector_store_id=vector_store_id
    )
    st.success("Vector Store ✓")
except Exception as e:
    st.error("Vector Store ✗")
    st.code(str(e))


# DeepSeek / Baseten
try:
    client_deepseek.models.list()
    st.success("DeepSeek ✓")
except Exception as e:
    st.error("DeepSeek ✗")
    st.code(str(e))
