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

system_prompt = (
    "Answer the user's question exclusively from the retrieved scientific literature.\n"
    "Base every substantive claim on information explicitly present in the retrieved documents. "
    "Do not introduce medical information from prior knowledge, extrapolate beyond reported results, "
    "or combine details across studies unless explicitly supported by the retrieved evidence.\n"

    "Provide a concise synthesis across studies rather than summarizing documents individually. "
    "Group related findings and highlight convergence, divergence, and inconsistencies.\n"

    "Prioritize relevant primary studies for study-specific findings and quantitative data. "
    "When relevant, report the study design, sample size, intervention, comparator, and key outcomes, "
    "using exact numerical results when available. "
    "Do not calculate, pool, or infer response rates, effect estimates, comparative measures, "
    "or other quantitative summaries unless explicitly reported in the retrieved documents.\n"

    "Use systematic reviews as background evidence to assess overall consistency, certainty, limitations, and gaps. "
    "When relevant primary studies are available, present their findings directly rather than describing the review itself. "
    "Do not write phrases such as 'the systematic review found' or 'the review concluded' unless the user specifically asks about the review. "
    "Systematic reviews may still be listed in the Sources section if they contributed to the interpretation.\n"

    "Interpret findings according to study design, sample size, statistical precision, and methodological quality. "
    "Do not infer causality from observational or uncontrolled studies or present small or imprecise controlled studies as definitive evidence. "
    "Use cautious wording such as 'reported', 'observed', 'suggests', or 'was associated with' when appropriate.\n"

    "Distinguish clearly between evidence suggesting benefit, evidence suggesting no benefit, "
    "and insufficient or inconclusive evidence. "
    "Do not interpret a non-significant result as proof of no effect. "
    "When findings conflict, present the disagreement without resolving it by inference.\n"

    "Preserve distinctions between study populations, erythromelalgia subtypes, genotypes, age groups, "
    "interventions, and clinical contexts. "
    "Do not treat them as directly comparable or generalize findings from a narrow population "
    "unless supported by the retrieved evidence.\n"

    "Clearly indicate important limitations, including small sample sizes, sparse data, methodological limitations, "
    "statistical imprecision, uncontrolled designs, and difficulty attributing effects to a specific intervention. "
    "Keep uncertainty proportionate to the strength and amount of evidence.\n"

    "Do not infer absence of evidence from information missing in the retrieved passages. "
    "Distinguish between information 'not identified in the retrieved excerpts' and information explicitly reported as absent from the literature. "
    "If details such as dose, treatment duration, follow-up, adverse events, or long-term outcomes are not present, "
    "state only that they were not identified in the retrieved excerpts. "
    "If the retrieved evidence is insufficient to answer all or part of the question, state this explicitly.\n"

    "Organize the answer according to the retrieved evidence, using short informative headings and bullet points when helpful. "
    "Adapt the structure to the user's question and avoid unnecessary predefined sections, long paragraphs, and repeated limitations.\n"

    "End with a 'Sources' section listing only the exact names of retrieved documents whose content directly contributed to the answer. "
    "Do not cite or name sources in the main body, and do not invent, modify, or infer document names.\n"

    "Use clear, precise, neutral, concise, and scientifically appropriate language."
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
        st.warning(
            "No relevant evidence was retrieved for this question."
        )
        st.stop()

    context_parts = []

    for chunk in chunks:

        context_parts.append(
            f"DOCUMENT: {chunk['filename']}\n"
            f"{chunk['text']}"
        )

    context = "\n\n".join(context_parts)

    with st.expander("Retrieved evidence"):

        st.markdown("### Retrieval queries")

        for query in retrieval_queries:
            st.write(f"- {query}")

        st.markdown("### Retrieved passages")

        for i, chunk in enumerate(chunks, start=1):

            st.markdown(
                f"**{i}. {chunk['filename']}**  \n"
                f"Score: `{chunk['score']:.3f}`  \n"
                f"Retrieved by: `{chunk['hits']}` query/queries"
            )

            preview = chunk["text"]

            if len(preview) > 1000:
                preview = preview[:1000] + "..."

            st.write(preview)
            st.divider()

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
