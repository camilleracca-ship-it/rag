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
                    "Answer the user's question exclusively using information explicitly supported by the retrieved documents. Do not introduce, infer, extrapolate, complete, or supplement information using prior model knowledge, even when the information is scientifically plausible or known from the literature.\n"
                    "For every study-specific or quantitative statement, ensure that the corresponding information is directly present in the retrieved context. Do not introduce study details, case reports, mutations, dosages, concomitant treatments, outcomes, or interpretations that are not explicitly stated in the retrieved documents. If a relevant detail is unavailable, state that it is not reported.\n"
                    "Provide a concise synthesis across studies, grouping related findings and highlighting areas of convergence, divergence, and inconsistency. When findings conflict, present the disagreement without attempting to resolve it through inference.\n"
                    "Prioritize relevant primary studies for study-specific findings and quantitative data. Report exact numerical results when available and, when relevant, include the study design, sample size, intervention, comparator, and key outcomes.\n"
                    "Use systematic reviews only as background evidence to help assess the overall consistency, certainty, limitations, and gaps in the literature. Do not describe or summarize the systematic review itself in the main body. Include it in the Sources section only if it directly contributed to the answer.\n"
                    "Do not calculate or infer pooled effects, response rates, comparative estimates, or other quantitative summaries unless they are explicitly reported in the retrieved documents.\n"
                    "Interpret findings in light of the study design, sample size, statistical precision, and methodological limitations. Do not infer causality from observational or uncontrolled studies, and do not present small, uncontrolled, or statistically imprecise studies as definitive evidence of efficacy.\n"
                    "Distinguish clearly between evidence suggesting benefit, evidence suggesting no benefit, and inconclusive or insufficient evidence. Do not interpret a non-significant result as evidence of no effect.\n"
                    "Preserve distinctions between study populations, erythromelalgia subtypes, genotypes, age groups, interventions, and clinical contexts. Do not treat findings from different populations or contexts as directly comparable unless the retrieved evidence supports such a comparison.\n"
                    "Clearly indicate when evidence is limited by small sample size, uncontrolled design, methodological limitations, statistical imprecision, sparse data, confounding, or difficulty attributing an observed effect to a specific intervention.\n"
                    "Explicitly identify gaps in the retrieved evidence. If the available documents do not contain sufficient information to answer all or part of the user's question, state this clearly rather than inferring an answer.\n"
                    "Organize the response according to the retrieved evidence, using short informative headings and bullet points when useful.\n"
                    "Use clear, precise, neutral, concise, and scientifically appropriate language. Do not mention author names, study names, publication years, filenames, or citation labels in the main body. Refer to individual studies only by their design or relevant characteristics when needed.\n"
                    "End with a "Sources" section listing only the exact names of retrieved documents that directly contributed to the answer.Answer the user's question exclusively using information explicitly present in the retrieved documents. Do not introduce, infer, or extrapolate beyond what those documents support.\n"
                 
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
