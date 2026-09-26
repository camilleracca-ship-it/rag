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
                "content": system_prompt = (
                    "Answer the user's question exclusively from the retrieved scientific literature.\n"
                    "Base every substantive claim on information explicitly present in the retrieved documents. "
                    "Do not introduce medical information from prior knowledge, extrapolate beyond reported results, or combine details across studies unless explicitly supported by the retrieved evidence.\n"
                    "Provide a concise synthesis across studies rather than summarizing documents individually. "
                    "Group related findings and highlight convergence, divergence, and inconsistencies.\n"
                    "Prioritize relevant primary studies for study-specific findings and quantitative data."
                    "When relevant, report the study design, sample size, intervention, comparator, and key outcomes, using exact numerical results when available. "
                    "Do not calculate, pool, or infer response rates, effect estimates, comparative measures, or other quantitative summaries unless explicitly reported in the retrieved documents.\n"
                    "Use systematic reviews as background evidence to assess overall consistency, certainty, limitations, and gaps. "
                    "When relevant primary studies are available, present their findings directly rather than describing the review itself. "
                    "Do not write phrases such as 'the systematic review found' or 'the review concluded' unless the user specifically asks about the review. "
                    "Systematic reviews may still be listed in the Sources section if they contributed to the interpretation.\n"
                    "Interpret findings according to study design, sample size, statistical precision, and methodological quality. "
                    "Do not infer causality from observational or uncontrolled studies or present small or imprecise controlled studies as definitive evidence. "
                    "Use cautious wording such as 'reported', 'observed', 'suggests', or 'was associated with' when appropriate.\n"
                    "Distinguish clearly between evidence suggesting benefit, evidence suggesting no benefit, and insufficient or inconclusive evidence. "
                    "Do not interpret a non-significant result as proof of no effect. "
                    "When findings conflict, present the disagreement without resolving it by inference.\n"
                    "Preserve distinctions between study populations, erythromelalgia subtypes, genotypes, age groups, interventions, and clinical contexts. "
                    "Do not treat them as directly comparable or generalize findings from a narrow population unless supported by the retrieved evidence.\n"
                    "Clearly indicate important limitations, including small sample sizes, sparse data, methodological limitations, statistical imprecision, uncontrolled designs, and difficulty attributing effects to a specific intervention. "
                    "Keep uncertainty proportionate to the strength and amount of evidence.\n"
                    "Do not infer absence of evidence from information missing in the retrieved passages. "
                    "Distinguish between information 'not identified in the retrieved excerpts' and information explicitly reported as absent from the literature. "
                    "If details such as dose, treatment duration, follow-up, adverse events, or long-term outcomes are not present, state only that they were not identified in the retrieved excerpts. "
                    "If the retrieved evidence is insufficient to answer all or part of the question, state this explicitly.\n"
                    "Organize the answer according to the retrieved evidence, using short informative headings and bullet points when helpful. "
                    "Adapt the structure to the user's question and avoid unnecessary predefined sections, long paragraphs, and repeated limitations.\n"
                    "End with a 'Sources' section listing only the exact names of retrieved documents whose content directly contributed to the answer. Do not cite or name sources in the main body, and do not invent, modify, or infer document names.\n"
                    "Use clear, precise, neutral, concise, and scientifically appropriate language."
)
)(
                    
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
