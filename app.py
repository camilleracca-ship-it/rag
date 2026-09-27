import streamlit as st
from openai import OpenAI
from study_metadata import study_metadata


st.title("Retrieval-Augmented Generation for Erythromelalgia")


client_openai = OpenAI(
    api_key=st.secrets["OPENAI_API_KEY"]
)

client_deepseek = OpenAI(
    api_key=st.secrets["BASETEN_API_KEY"],
    base_url="https://inference.baseten.co/v1"
)

vector_store_id = st.secrets["OPENAI_VECTOR_STORE_ID"]


system_prompt = (
    "Answer the user's question exclusively from the retrieved scientific excerpts and the provided study metadata. "
    "Do not introduce medical information from prior knowledge or extrapolate beyond the reported evidence.\n"

    "Provide a concise synthesis across studies rather than summarizing studies one by one. "
    "Focus only on evidence that directly helps answer the user's question and do not provide a comprehensive review of the retrieved literature. "
    "Begin with a brief overview of the evidence and limitations most relevant to the user's question. "
    "Then present first the evidence that most closely matches the user's question and, when provided, the clinical context. "
    "Within evidence of similar relevance, present clinical trials and prospective studies before retrospective and other observational studies. "
    "Less directly applicable evidence should be mentioned only briefly when it helps contextualize the answer. "
    "Preserve distinctions between populations, disease subtypes, genotypes, age groups, interventions, and clinical contexts. "
    "Do not generalize findings unless explicitly supported by the retrieved evidence.\n"

    "Report study design, sample size, participant characteristics, intervention, comparator, outcomes, or mechanistic findings "
    "only when they are needed to answer the user's question or evaluate the strength and limitations of the evidence. "
    "Do not systematically report these elements for every study. "
    "Report only outcomes directly relevant to the user's question, including efficacy, safety, or mechanistic outcomes when relevant. "
    "When a directly relevant outcome is reported as a primary or secondary outcome, clearly identify its status. "
    "Do not calculate, pool, or infer response rates, effect estimates, or other quantitative summaries that are not explicitly reported.\n"

    "Report study-specific findings and quantitative results exclusively from the retrieved primary studies. "
    "Use systematic reviews only to assess the consistency, certainty, limitations, and gaps in the evidence. "
    "Incorporate these considerations into the synthesis without presenting the systematic review as a separate body of evidence or using it as a source of study-specific results.\n"

    "Interpret findings according to study design, sample size, statistical precision, methodological quality, and publication type. "
    "Do not infer causality from observational or uncontrolled studies or treat small or imprecise controlled studies as definitive evidence. "
    "Consider publication status when interpreting the evidence. Distinguish results published in peer-reviewed journals from results available only in clinical trial registries, and consider the absence of peer review as a limitation when relevant."

    "Distinguish evidence suggesting benefit, evidence suggesting no benefit, and insufficient or inconclusive evidence. "
    "Do not interpret a non-significant result as proof of no effect or resolve conflicting findings through unsupported inference. "
    "Use cautious wording where appropriate and keep uncertainty proportionate to the strength and amount of evidence. "
    "Explicitly acknowledge when the retrieved excerpts lack requested details or provide insufficient evidence to answer all or part of the question.\n"

    "Include a short limitations section containing only the main limitations that materially affect interpretation of the answer. "
    "Avoid repeating limitations or findings already stated in the main synthesis.\n"

    "Respond in the user's language with clear, precise, neutral, concise, and scientifically appropriate wording. "
    "Adapt the structure to the question, using short informative headings and bullet points when helpful.\n"

    "List sources only in a final 'Sources' section, including only publications whose retrieved content contributed to the answer or its interpretation. "
    "Use the author, article title, and publication date exactly as provided in the metadata."
)


def build_retrieval_queries(question):
    q = question.strip()

    queries = [
        q,

        (
            f"{q} "
            "primary study clinical trial prospective retrospective cohort case series "
            "sample size quantitative results treatment response efficacy"
        ),

        (
            f"{q} "
            "primary secondary acquired idiopathic hereditary erythromelalgia eryththermalgia "
            "etiology cause associated disease autoimmune hematologic neurological"
        ),

        (
            f"{q} "
            "pediatric paediatric child children adolescent juvenile adult age onset"
        ),

        (
            f"{q} "
            "small fiber neuropathy small fibre neuropathy SFN "
            "intraepidermal nerve fiber density IENFD skin biopsy autonomic sensory neuropathy"
        ),

        (
            f"{q} "
            "SCN9A Nav1.7 sodium channel mutation variant genotype phenotype hereditary genetic"
        ),

        (
            f"{q} "
            "dose dosage administration titration treatment duration follow-up "
            "long-term outcome adverse events safety tolerability discontinuation"
        )
    ]

    return list(dict.fromkeys(queries))


def retrieve_evidence(question):
    retrieval_queries = build_retrieval_queries(question)

    all_results = []

    for retrieval_query in retrieval_queries:
        results = client_openai.vector_stores.search(
            vector_store_id=vector_store_id,
            query=retrieval_query,
            max_num_results=15,
            rewrite_query=True
        )

        all_results.extend(results.data)

    unique_results = {}

    for result in all_results:
        for content in result.content:
            if content.type != "text":
                continue

            normalized_text = " ".join(content.text.split())

            key = (
                result.filename,
                normalized_text
            )

            if key not in unique_results:
                unique_results[key] = {
                    "filename": result.filename,
                    "text": content.text,
                    "score": result.score,
                    "hits": 1
                }

            else:
                unique_results[key]["hits"] += 1

                if result.score > unique_results[key]["score"]:
                    unique_results[key]["score"] = result.score

    ranked_chunks = sorted(
        unique_results.values(),
        key=lambda chunk: (
            chunk["hits"],
            chunk["score"]
        ),
        reverse=True
    )

    selected_chunks = []
    chunks_per_file = {}

    max_chunks_per_file = 4
    max_total_chunks = 30

    for chunk in ranked_chunks:
        if len(selected_chunks) >= max_total_chunks:
            break

        filename = chunk["filename"]

        if chunks_per_file.get(filename, 0) >= max_chunks_per_file:
            continue

        selected_chunks.append(chunk)

        chunks_per_file[filename] = (
            chunks_per_file.get(filename, 0) + 1
        )

    return selected_chunks, retrieval_queries


question = st.text_input(
    "Ask a question about pain management in erythromelalgia"
)

if question:
    with st.spinner("Searching the scientific literature..."):
        chunks, retrieval_queries = retrieve_evidence(question)

    if not chunks:
        st.warning("No relevant evidence was retrieved for this question.")
        st.stop()

    context_parts = []

    for chunk in chunks:
        metadata = study_metadata.get(chunk["filename"], {})

        metadata_text = "\n".join(
            f"{key}: {value}"
            for key, value in metadata.items()
        )

        context_parts.append(
            f"DOCUMENT: {chunk['filename']}\n"
            f"STUDY METADATA:\n{metadata_text}\n\n"
            f"RETRIEVED EXCERPT:\n{chunk['text']}"
        )

    context = "\n\n".join(context_parts)

    with st.spinner("Synthesizing the evidence..."):
        response = client_deepseek.chat.completions.create(
            model="deepseek-ai/DeepSeek-V4.1-Flash",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
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
