import streamlit as st
from openai import OpenAI
from study_metadata import study_metadata

locale = st.context.locale or "en"
language = "fr" if locale.lower().startswith("fr") else "en"

translations = {
    "en": {
        "title": "Pain Evidence",
        "subtitle": "Evidence-based answers from the scientific literature using Retrieval-Augmented Generation (RAG).",
        "domain": "Evidence domain",
        "question": "Ask your question...",
        "searching": "Searching the scientific literature... This may take a few minutes.",
        "synthesizing": "Synthesizing the evidence... Thank you for your patience.",
        "no_evidence": "No relevant evidence was retrieved for this question.",
        "response_language": "English"
    },
    "fr": {
        "title": "Pain Evidence",
        "subtitle": "Des réponses fondées sur la littérature scientifique grâce au Retrieval-Augmented Generation (RAG).",
        "domain": "Domaine de connaissances",
        "question": "Posez votre question...",
        "searching": "Recherche dans la littérature scientifique... Cela peut prendre quelques minutes.",
        "synthesizing": "Synthèse des données... Merci pour votre patience.",
        "no_evidence": "Aucune donnée pertinente n’a été retrouvée pour cette question.",
        "response_language": "French"
    }
}

t = translations[language]
response_language = t["response_language"]

st.title(t["title"])
st.markdown(t["subtitle"])

domain_labels = {
    "erythromelalgia": {
        "en": "Erythromelalgia",
        "fr": "Érythromélalgie"
    }
}

domain = st.selectbox(
    t["domain"],
    ["erythromelalgia"],
    format_func=lambda x: domain_labels[x][language]
)

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
    "Do not introduce medical information from prior knowledge or extrapolate beyond the reported evidence. "
    "If the retrieved evidence is insufficient to answer all or part of the question, state this clearly.\n"

    "Preserve distinctions between populations, disease subtypes, genotypes, age groups, interventions, and clinical contexts. "
    "If a finding is reported only for a specific subgroup or context, state this explicitly and do not generalise it beyond that subgroup or context unless supported by the evidence. "

    "When a disease subtype is specified, prioritize evidence from that subtype. "
    "When disease subtype is not specified, report findings for each disease subtype represented in the retrieved evidence, clearly identifying the subtype, and do not generalise findings from one subtype to others unless supported by the evidence. "
    "If no relevant data are available for a particular subtype, state this explicitly. "

    "Prioritize evidence that more closely matches the characteristics explicitly specified in the user's question, including disease subtype, genotype, age group, intervention, and clinical context. "
    "Maintain this priority even when such evidence is not predominant in the retrieved material.\n"

    "Provide a concise synthesis across studies rather than summarizing studies one by one. "
    "When relevance is comparable, present first findings supported by larger samples and prospective or controlled study designs. "
    "Evidence with lower direct relevance should be mentioned only if it helps contextualize or interpret the answer. "
    "When included, it should be kept brief.\n"

    "When participants receive concomitant treatments, preserve this context and report outcomes for the treatment regimen as a whole, without attributing them to a single intervention unless supported by the study design. "

    "Report only outcomes and underlying mechanisms that are relevant to the user's question or necessary to interpret the evidence. "
    "Report underlying mechanisms only when they are explicitly described in the retrieved evidence. "
    "For relevant outcomes, indicate whether they were primary or secondary only when explicitly reported; do not infer this classification. "
    "Do not calculate, pool, or infer response rates, effect estimates, or other quantitative summaries that are not explicitly reported.\n"

    "Use appropriate caution when interpreting observational or uncontrolled studies, small or imprecise controlled studies, secondary outcomes, and registry-only results. "
    "Do not infer causality from observational evidence or overstate conclusions from limited evidence. "
    "Keep uncertainty proportionate to the strength, amount, and consistency of the evidence. "
    "Distinguish evidence suggesting benefit, evidence explicitly suggesting no benefit, non-significant or inconclusive findings, and insufficient evidence. "
    "Do not interpret a non-significant result as evidence of no effect or resolve conflicting findings through unsupported inference.\n"

    "Use systematic reviews only to contextualize the overall consistency, certainty, limitations, and gaps in the evidence. "
    "Use primary studies for study-specific findings and quantitative results. "
    "Include only the main limitations that materially affect interpretation, without repeating limitations already stated elsewhere in the answer.\n"

    "Begin with a brief summary of the overall evidence and its main limitations. "
    "Structure the response naturally around the user's question, using short informative headings and bullet points when helpful. "
    "Respond in the language specified in the USER INTERFACE LANGUAGE field provided with the question, using clear, precise, neutral, concise, and scientifically appropriate wording.\n"

    "List sources only in a final sources section, with the section heading in the response language, including only publications whose retrieved content contributed to the answer or its interpretation. "
    "Do not cite or name sources in the main body of the answer. "
    "Use the author, article title, and publication date exactly as provided in the metadata."
)

def build_retrieval_queries(question):
    q = question.strip()

    queries = [
        q,

        (
            f"{q} "
            "systematic review primary study clinical trial "
            "randomized controlled trial randomised controlled trial "
            "prospective retrospective cohort case series "
            "sample size quantitative results treatment response efficacy"
        ),

        (
            f"{q} "
            "primary secondary acquired idiopathic hereditary erythromelalgia eryththermalgia "
            "etiology cause associated disease autoimmune hematologic myeloproliferative neurological"
        ),

        (
            f"{q} "
            "pediatric paediatric child children adolescent juvenile adult age onset"
        ),

        (
            f"{q} "
            "small fiber neuropathy small fibre neuropathy "
            "skin biopsy autonomic sensory neuropathy"
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

    unique_results = {}
    per_query_candidates = []

    for retrieval_query in retrieval_queries:
        results = client_openai.vector_stores.search(
            vector_store_id=vector_store_id,
            query=retrieval_query,
            max_num_results=30,
            rewrite_query=True
        )

        query_keys = []
        seen_in_this_query = set()

        for rank, result in enumerate(results.data, start=1):
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
                        "hits": 0,
                        "matched_queries": [],
                        "query_ranks": {}
                    }

                if result.score > unique_results[key]["score"]:
                    unique_results[key]["score"] = result.score

                if key not in seen_in_this_query:
                    unique_results[key]["hits"] += 1
                    unique_results[key]["matched_queries"].append(
                        retrieval_query
                    )
                    unique_results[key]["query_ranks"][
                        retrieval_query
                    ] = rank

                    query_keys.append(key)
                    seen_in_this_query.add(key)

        per_query_candidates.append({
            "query": retrieval_query,
            "keys": query_keys
        })

    ranked_chunks = sorted(
        unique_results.values(),
        key=lambda chunk: (
            chunk["hits"],
            chunk["score"]
        ),
        reverse=True
    )

    selected_chunks = []
    selected_keys = set()
    chunks_per_file = {}

    min_chunks_per_query = 4
    max_chunks_per_file = 5
    max_total_chunks = 30

    chunks_added_per_query = [
        0 for _ in per_query_candidates
    ]

    query_positions = [
        0 for _ in per_query_candidates
    ]

    while len(selected_chunks) < max_total_chunks:
        progress = False

        for i, query_data in enumerate(per_query_candidates):
            if len(selected_chunks) >= max_total_chunks:
                break

            if chunks_added_per_query[i] >= min_chunks_per_query:
                continue

            keys = query_data["keys"]

            while query_positions[i] < len(keys):
                key = keys[query_positions[i]]
                query_positions[i] += 1

                if key in selected_keys:
                    continue

                chunk = unique_results[key]
                filename = chunk["filename"]

                if chunks_per_file.get(filename, 0) >= max_chunks_per_file:
                    continue

                selected_chunks.append(chunk)
                selected_keys.add(key)

                chunks_per_file[filename] = (
                    chunks_per_file.get(filename, 0) + 1
                )

                chunks_added_per_query[i] += 1
                progress = True
                break

        if not progress:
            break

        if all(
            n >= min_chunks_per_query
            for n in chunks_added_per_query
        ):
            break

    for chunk in ranked_chunks:
        if len(selected_chunks) >= max_total_chunks:
            break

        filename = chunk["filename"]

        chunk_key = (
            chunk["filename"],
            " ".join(chunk["text"].split())
        )

        if chunk_key in selected_keys:
            continue

        if chunks_per_file.get(filename, 0) >= max_chunks_per_file:
            continue

        selected_chunks.append(chunk)
        selected_keys.add(chunk_key)

        chunks_per_file[filename] = (
            chunks_per_file.get(filename, 0) + 1
        )

    return selected_chunks, retrieval_queries, ranked_chunks


question = st.text_area(
    t["question"],
    placeholder=t["question_placeholder"],
    height=120,
    label_visibility="collapsed"
)

if question:
    with st.spinner(t["searching"]):
        chunks, retrieval_queries, ranked_chunks = retrieve_evidence(question)

    if not chunks:
        st.warning(t["no_evidence"])
        st.stop()

    context_parts = []

    for chunk in chunks:
        metadata = study_metadata.get(
            chunk["filename"],
            {}
        )

        metadata_text = "\n".join(
            f"{key}: {value}"
            for key, value in metadata.items()
        )

        context_parts.append(
            f"DOCUMENT: {chunk['filename']}\n"
            f"STUDY METADATA:\n"
            f"{metadata_text}\n\n"
            f"RETRIEVED EXCERPT:\n"
            f"{chunk['text']}"
        )

    context = "\n\n".join(context_parts)

    with st.spinner(t["synthesizing"]):
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
                        f"USER INTERFACE LANGUAGE: {response_language} ({locale})\n\n"
                        f"Question:\n{question}\n\n"
                        f"Retrieved scientific literature:\n"
                        f"{context}"
                    )
                }
            ]
        )

    answer = response.choices[0].message.content
    st.markdown(answer)
