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

question = st.text_input(
    "Ask a question about pain management in erythromelalgia"
)

if question:

    results = client_openai.vector_stores.search(
        vector_store_id=vector_store_id,
        query=question
    )

    context_parts = []

    for result in results.data:
        for content in result.content:
            if content.type == "text":
                context_parts.append(
                    f"DOCUMENT: {result.filename}\n"
                    f"{content.text}"
                )

    context = "\n\n".join(context_parts)

    response = client_deepseek.chat.completions.create(
        model="deepseek-ai/DeepSeek-V4.1-Flash",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a scientific assistant specialized in pain management in erythromelalgia. "
                    "Answer the user's question exclusively from the retrieved scientific literature. "
                    "Provide a concise synthesis of the evidence across studies rather than summarizing each document separately. "
                    "Report the main findings accurately and, when relevant, include the study design, sample size, intervention, comparator, and key outcomes. "
                    "Preserve important quantitative results when they are available in the retrieved documents. "
                    "Clearly identify differences or inconsistencies between studies and indicate when the evidence is limited by small sample sizes, methodological limitations, or sparse data. "
                    "Do not extrapolate beyond the results reported in the retrieved documents. "
                    "Do not introduce any medical information that is not explicitly supported by the retrieved documents. "
                    "Interpret associations, treatment effects, and causal relationships in accordance with the study design, sample size, statistical precision, and overall methodological quality of the retrieved studies. "
                    "Do not present observational or uncontrolled findings as evidence of causality, and do not present findings from small or imprecise controlled trials as definitive evidence of treatment efficacy. "
                    "Do not treat different study populations, erythromelalgia subtypes, or clinical contexts as directly comparable unless the retrieved evidence supports such a comparison. "
                    "Distinguish between evidence of benefit, evidence of no benefit, and insufficient or inconclusive evidence. "
                    "Do not interpret a non-significant result as proof of no effect. "
                    "At the end of the response, add a 'Sources' section listing only the names of the retrieved documents that were actually used to support the answer. "
                    "Do not include citations or source names in the main body of the response. "
                    "Do not list a document in the Sources section unless its retrieved content contributed directly to the answer. "
                    "Do not invent, modify, or infer source names; use only the document names provided in the retrieved context. "
                    "If the retrieved literature is insufficient to answer the question, state this explicitly. "
                    "Use clear, precise, neutral, and concise scientific language."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\n"
                    f"Retrieved scientific literature:\n{context}"
                )
            }
        ]
    )

    answer = response.choices[0].message.content

    st.write(answer)

if question:
    results = client_openai.vector_stores.search(
        vector_store_id=vector_store_id,
        query=question
    )

    st.write("Number of retrieved results:", len(results.data))
    st.write(results)
