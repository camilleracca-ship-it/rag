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

from pathlib import Path

dossier_articles = Path(__file__).parent / "data"

fichiers_pdf = list(dossier_articles.glob("*.pdf"))
vector_store_id = st.secrets["OPENAI_VECTOR_STORE_ID"]

