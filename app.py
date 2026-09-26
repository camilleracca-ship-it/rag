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

    if not context:
        st.warning(
            "No relevant evidence was retrieved for this question."
        )
        st.stop()

    response = client_deepseek.chat.completions.create(
        model="deepseek-ai/DeepSeek-V4.1-Flash",
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer the user's question exclusively using information explicitly present in the retrieved documents. Do not introduce, infer, or extrapolate beyond what those documents support.\n"
                    "Provide a concise synthesis across studies, grouping related findings and highlighting areas of convergence, divergence, and inconsistency. When findings conflict, present the disagreement without resolving it by inference.\n"
                    "Prioritize relevant primary studies for study-specific findings and quantitative data. Report exact numbers when available and, when relevant, include the study design, sample size, intervention, comparator, and key outcomes.\n"
                    "Use systematic reviews as background evidence to help interpret the overall consistency, certainty, and limitations of the literature. Do not describe, summarize, or discuss the systematic review itself in the main body of the answer. Include systematic reviews in the Sources section only if they contributed to the answer.\n"
                    "Do not calculate or infer pooled effects, response rates, comparative estimates, or other quantitative summaries unless explicitly reported in the retrieved documents.\n"
                    "Interpret findings according to the study design, sample size, statistical precision, and methodological quality. Do not infer causality from observational or uncontrolled studies, or present small or imprecise controlled trials as definitive evidence of efficacy.\n"
                    "Distinguish between evidence of benefit, evidence of no benefit, and insufficient or inconclusive evidence. Do not interpret a non-significant result as proof of no effect.\n"
                    "Preserve distinctions between study populations, erythromelalgia subtypes, genotypes, age groups, and clinical contexts, and do not treat them as directly comparable unless supported by the retrieved evidence.\n"
                    "Clearly indicate when the evidence is limited by small sample sizes, methodological limitations, statistical imprecision, sparse data, uncontrolled designs, or difficulty attributing effects to a specific intervention.\n"
                    "Explicitly identify gaps in the retrieved evidence and state when the available evidence is insufficient to answer the user's question.\n"
                    "Organize the response according to the retrieved evidence, using short informative headings and bullet points when helpful.\n"
                    "Use clear, precise, neutral, concise, and scientifically appropriate language. Do not mention author names, study names, publication years, filenames, or citation labels in the main body; refer to studies only by their design or characteristics when needed. End with a 'Sources' section listing only the exact names of retrieved documents that directly contributed to the answer.\n"
                 
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

    st.markdown(answer)
