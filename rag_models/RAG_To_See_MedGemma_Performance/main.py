# main.py
import re
import os, tempfile
import time
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util
from bert_score import score

from indexing import setup_vector_store
from prompt import get_discharge_summary_prompt, get_llm

# ==============================
# CONFIGURATION
# ==============================
load_dotenv()

# Use only HF_HOME going forward
os.environ["HF_HOME"] = ".cache"
os.environ["TORCH_HOME"] = ".cache/torch"
os.environ["TMPDIR"] = ".cache/temp"
tempfile.tempdir = ".cache/temp"

DATA_PATH = "data/MIMIC_NOTE.txt"
CHROMA_PATH = "vectorDB/chroma_test/"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
RESULT_FILE = os.path.join(RESULTS_DIR, "evaluation.txt")


# ==============================
# INDEXING
# ==============================
vector_store = setup_vector_store(DATA_PATH, CHROMA_PATH)

def get_precise_retriever(vector_store):
    """
    Returns a retriever optimized for disease-specific clinical summarization.
    Uses high-similarity retrieval to ensure COPD-focused context.
    """
    retriever = vector_store.as_retriever(
                search_type="mmr",
                # search_kwargs={"k": 20, "fetch_k": 100, "lambda_mult": 0.5}
                search_kwargs={"k": 12, "fetch_k": 70, "lambda_mult": 0.45} # for chroma_test
    )
    return retriever

retriever = get_precise_retriever(vector_store)


# vector_store = setup_vector_store(DATA_PATH, CHROMA_PATH)
# retriever = vector_store.as_retriever(
#     search_type="mmr",
#     search_kwargs={"k": 10, "fetch_k": 20, "lambda_mult": 0.7}
# )
# retriever = vector_store.as_retriever(
#     search_type="similarity",
#     search_kwargs={"k": 8})

# ==============================
# PROMPT + MODEL
# ==============================
prompt = get_discharge_summary_prompt()
llm = get_llm()

# ==============================
# COHERENCE MODELS
# ==============================
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# ==============================
# RAG Graph Definition
# ==============================
class RAGState(dict):
    question: str
    context: str
    answer: str

RETRIEVED_FILE = os.path.join("results", "retrieved_chunks.txt")


def retrieve(state: RAGState):
    start = time.time()
    query = state["question"].lower()

    # ---- Pass A: disease-focused (original query) ----
    docs_a = retriever.invoke(query)

    # ---- Pass B: structure-enriched query ----
    # remove "discharge" to avoid bias toward tail sections
    structure_query = re.sub(r"discharge", "", query)
    structure_query += (
        " history of present illness, hospital course, physical examination, "
        "labs, imaging, medications, and follow-up details"
    )
    docs_b = retriever.invoke(structure_query)

    # ---- Combine and de-duplicate ----
    all_docs = docs_a + docs_b
    unique_docs = []
    seen = set()
    for doc in all_docs:
        if doc.page_content not in seen:
            unique_docs.append(doc)
            seen.add(doc.page_content)

    # ---- Re-rank by section coverage ----
    section_terms = [
        "history", "hospital course", "physical", "exam", "lab", "imaging",
        "chronic", "transition", "medication", "discharge", "instruction"
    ]

    scored = []
    for doc in unique_docs:
        text = doc.page_content.lower()
        section_hits = sum(s in text for s in section_terms)
        score = section_hits
        scored.append((score, doc))

    # sort and keep top 14 diverse chunks
    ranked = [d for s, d in sorted(scored, key=lambda x: x[0], reverse=True)][:14]

    # merge and save
    state["context"] = "\n\n".join(doc.page_content.strip() for doc in ranked)

    print(f"✅ Retrieved {len(ranked)} chunks (hybrid search).")
    print(f"📄 Context length: {len(state['context'].split())} words")

    with open(RETRIEVED_FILE, "w", encoding="utf-8") as f:
        for i, doc in enumerate(ranked):
            f.write(f"--- Chunk {i+1} ---\n{doc.page_content}\n\n")

    state["retrieval_time"] = time.time() - start
    print(f"⏱️ Retrieval took {state['retrieval_time']:.2f} s")
    return state



def generate(state: RAGState):
    start = time.time()
    chain = prompt | llm
    state["answer"] = chain.invoke({
        "context": state["context"],
        "question": state["question"]
    })
    print(f"⏱️ Generation took {time.time() - start:.2f} seconds")
    return state

graph = StateGraph(RAGState)
graph.add_node("retrieve", retrieve)
graph.add_node("generate", generate)
graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)
app = graph.compile()

# ==============================
# RUN PIPELINE
# ==============================
query = (
    "Generate a full hospital discharge summary including "
    "admission history, hospital course, physical exam, labs, "
    "imaging, medications, and discharge instructions "
    "for a patient with COPD and comorbid hypertension and hyperlipidemia."
)
result = app.invoke({"question": query})

generated_text = result["answer"].content

# ==============================
# PHASE 3: EVALUATION
# ==============================
print("🔹 Starting Phase 3: Evaluation")
start_time = time.time()

REFERENCE_PATH = os.path.join("data", "reference.txt")

if not os.path.exists(REFERENCE_PATH):
    raise FileNotFoundError(f"⚠️ Reference file not found at {REFERENCE_PATH}.")

with open(REFERENCE_PATH, "r", encoding="utf-8") as ref_file:
    reference_text = ref_file.read().strip()

generated_text = result["answer"].content

# Step 3b. Compute BLEU
bleu_score = sentence_bleu([reference_text.split()], generated_text.split())

# Step 3c. Compute ROUGE-L
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
rouge_l = scorer.score(reference_text, generated_text)["rougeL"].fmeasure

# Step 3d. Compute SBERT Coherence
ref_emb = sbert_model.encode(reference_text, convert_to_tensor=True)
gen_emb = sbert_model.encode(generated_text, convert_to_tensor=True)
sbert_coherence = util.cos_sim(ref_emb, gen_emb).item()

# Step 3e. Compute BERTScore
P, R, F1 = score([generated_text], [reference_text], lang="en", verbose=False)
bert_f1 = F1.mean().item()

# Step 3f. Save all metrics
with open(RESULT_FILE, "w", encoding="utf-8") as f:
    f.write("=== Evaluation Metrics ===\n")
    f.write(f"User Query: {result.get('question', 'N/A')}\n\n")
    f.write(f"BLEU Score: {bleu_score:.4f}\n")
    f.write(f"ROUGE-L Score: {rouge_l:.4f}\n")
    f.write(f"SBERT Coherence: {sbert_coherence:.4f}\n")
    f.write(f"BERTScore F1 (Semantic Coherence): {bert_f1:.4f}\n")
    f.write(f"Retrieval Time: {result.get('retrieval_time', 0):.2f} sec\n")
    f.write(f"Generation Time: {result.get('generation_time', 0):.2f} sec\n\n")
    f.write("--- Generated Summary ---\n")
    f.write(generated_text)

print(f"✅ Evaluation completed — saved to {RESULT_FILE}")

