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

file_streams = [
    open(fichier, "rb")
    for fichier in fichiers_pdf
]

client.vector_stores.file_batches.upload_and_poll(
    vector_store_id=vector_store.id,
    files=file_streams
)

print("Vector Store prêt :", vector_store.id)
