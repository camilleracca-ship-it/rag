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
    
    "Provide a concise synthesis across studies rather than summarizing studies one by one.\n"
    "Preserve distinctions between populations, disease subtypes, genotypes, age groups, interventions, and clinical contexts. "
    "Evidence should be considered more directly relevant when these characteristics more closely match the user's question. "
    "Focus primarily on the evidence that is most directly relevant to the user's question. For example, when a disease subtype is specified in the user's question or clinical context, prioritize evidence relating to that subtype. "
    "When relevance is comparable, present first findings supported by larger samples and prospective or controlled study designs. "
    "When disease subtype is not specified in the user's question, treat the question as applying to the condition broadly. Report findings for each disease subtype represented in the retrieved evidence, and do not generalise findings from one subtype to others unless supported by the evidence. "
    "When participants receive concomitant treatments, preserve this context and report outcomes for the treatment regimen as a whole, without attributing them to a single intervention unless supported by the study design. "
    "Evidence with lower direct relevance should be mentioned only if it helps contextualize or interpret the answer. When included, it should be kept brief.\n"

    "Begin with a brief summary of the overall evidence and its main limitations. "
    "Structure the response naturally around the user's question, using short informative headings and bullet points when helpful. "
    "Do not generalize findings unless explicitly supported by the retrieved evidence.\n"
    
    "Report only outcomes and underlying mechanisms that are relevant to the user's question or necessary to interpret the evidence. "
    "For relevant outcomes, indicate whether they were primary or secondary."
    "Do not calculate, pool, or infer response rates, effect estimates, or other quantitative summaries that are not explicitly reported.\n"

    "Use appropriate caution when interpreting observational or uncontrolled studies, small or imprecise controlled studies, secondary outcomes, and registry-only results. "
    "Do not infer causality from observational evidence or overstate conclusions from limited evidence. "
    "Keep uncertainty proportionate to the strength, amount, and consistency of the evidence. "
    "Distinguish evidence suggesting benefit, evidence suggesting no benefit, and insufficient or inconclusive evidence. "
    "Do not interpret a non-significant result as evidence of no effect or resolve conflicting findings through unsupported inference. "
    "Clearly state when the retrieved evidence is insufficient to answer all or part of the user's question.\n"

    "Use systematic reviews only to contextualize the overall consistency, certainty, limitations, and gaps in the evidence. Use primary studies for study-specific findings and quantitative results."
    "Include only the main limitations that materially affect interpretation, without repeating limitations already stated elsewhere in the answer.\n"

    "Respond in the user's language using clear, precise, neutral, concise, and scientifically appropriate wording.\n"

    "List sources only in a final 'Sources' section, including only publications whose retrieved content contributed to the answer or its interpretation. "
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


question = st.text_input(
    "Ask a question about pain management in erythromelalgia"
)

if question:
    with st.spinner("Searching the scientific literature..."):
        chunks, retrieval_queries, ranked_chunks = retrieve_evidence(question)

    if not chunks:
        st.warning(
            "No relevant evidence was retrieved for this question."
        )
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

    with st.expander("Retrieval diagnostics"):

        selected_ids = {
            (
                chunk["filename"],
                " ".join(chunk["text"].split())
            )
            for chunk in chunks
        }

        for rank, chunk in enumerate(
            ranked_chunks,
            start=1
        ):
            chunk_id = (
                chunk["filename"],
                " ".join(chunk["text"].split())
            )

            selected = chunk_id in selected_ids

            metadata = study_metadata.get(
                chunk["filename"],
                {}
            )

            st.write(
                f"Rank {rank}",
                "|", chunk["filename"],
                "| hits:", chunk["hits"],
                "| score:", round(chunk["score"], 3),
                "| study design:",
                metadata.get("study_design", "Unknown"),
                "| SENT:",
                "YES" if selected else "NO"
            )

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
                        f"Retrieved scientific literature:\n"
                        f"{context}"
                    )
                }
            ]
        )

    answer = response.choices[0].message.content
    st.markdown(answer)
