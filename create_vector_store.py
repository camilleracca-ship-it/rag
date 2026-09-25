import streamlit as st
from openai import OpenAI
from pathlib import Path

client = OpenAI(
    api_key=st.secrets["OPENAI_API_KEY"]
)

dossier_articles = Path(__file__).parent / "data"

fichiers_pdf = list(dossier_articles.glob("*.pdf"))

vector_store = client.vector_stores.create(
    name="erythromelalgia_articles"
)

print("Vector Store créé :", vector_store.id)
