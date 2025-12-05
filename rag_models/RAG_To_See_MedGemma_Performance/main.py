# main.py
import re
import os, tempfile
os.environ["HF_HOME"] = "D:/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "D:/hf_cache"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "D:/hf_cache"
os.environ["TORCH_HOME"] = "D:/hf_cache"
os.environ["XDG_CACHE_HOME"] = "D:/hf_cache"
os.environ["TMPDIR"] = "D:/hf_cache/temp"
tempfile.tempdir = "D:/hf_cache/temp"
import time
import requests
import json
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util
from bert_score import score

# from indexing_schema import setup_schema_vector_store
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from prompt import get_discharge_summary_prompt

# ==============================
# CONFIGURATION
# ==============================
load_dotenv()

PATIENT_NAME = "Rakesh" 
# Options: "Toma", "Taylor", "Heath", "Nicholas", "Rakesh", "Bhavana"

TRANSCRIPT_PATH = f"data/transcription_{PATIENT_NAME.lower()}.txt"
OPENEMR_PATH = f"data/openemr_{PATIENT_NAME.lower()}.txt"
REFERENCE_PATH = f"data/reference_{PATIENT_NAME.lower()}.txt"

DATA_PATH = "data/all_notes_structure.json"
CHROMA_PATH = "vectorDB/chroma_schema_improved/"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
RESULT_FILE = os.path.join(RESULTS_DIR, f"evaluation_{PATIENT_NAME}.txt")

# ==============================
# INDEXING
# ==============================
# vector_store = setup_schema_vector_store(DATA_PATH, CHROMA_PATH)
print(f"🔹 Loading Vector Store from: {CHROMA_PATH}")
embeddings = HuggingFaceEmbeddings(
    model_name="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
)
vector_store = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

def get_precise_retriever(vector_store):
    """
    Returns a retriever optimized for disease-specific clinical summarization.
    """
    retriever = vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 12, "fetch_k": 70, "lambda_mult": 0.45} 
    )
    return retriever

retriever = get_precise_retriever(vector_store)

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
    retrieval_time: float
    generation_time: float
    input_tokens: int
    output_tokens: int
    total_tokens: int

RETRIEVED_FILE = os.path.join("results", "retrieved_chunks.txt")

def extract_disease_from_transcript(transcript: str, llm_instance) -> str:
    """
    Uses the LLM to extract the primary chronic condition or reason for admission
    from the transcript.
    """
    print("DEBUG: Extracting disease from transcript...")
    extraction_prompt = (
        "You are a medical expert. Analyze the following doctor-patient transcript "
        "and identify the PRIMARY chronic condition or main reason for the visit. "
        "Output ONLY the disease name (e.g., 'Diabetes', 'Hypertension', 'COPD'). "
        "Do not output a sentence.\n\n"
        f"TRANSCRIPT:\n{transcript[:2000]}...\n\n"  # Truncate for speed
        "DISEASE:"
    )
    
    try:
        disease = llm_instance.invoke(extraction_prompt).strip()
        disease = re.sub(r"^The disease is\s+", "", disease, flags=re.IGNORECASE)
        disease = disease.strip(" .")
        print(f"✅ Extracted Disease: {disease}")
        return disease
    except Exception as e:
        print(f"❌ Disease extraction failed: {e}")
        return "Unknown"

def retrieve(state: RAGState):
    start = time.time()
    
    # 1. Read Transcript
    try:
        with open(TRANSCRIPT_PATH, "r", encoding="utf-8") as f:
            transcript_text = f.read().strip()
    except Exception as e:
        print(f"⚠️ Could not read transcript from {TRANSCRIPT_PATH}: {e}")
        transcript_text = ""

    # 2. Extract Disease
    target_disease = extract_disease_from_transcript(transcript_text, llm)
    
    # 3. Semantic Metadata Matching
    print("DEBUG: Fetching metadata from Vector Store...")
    try:
        collection_data = vector_store.get(include=["metadatas"])
        all_metadatas = collection_data["metadatas"]
        all_ids = collection_data["ids"]
        
        if not all_metadatas:
            print("❌ No metadata found in vector store.")
            state["context"] = ""
            return state

        # Prepare for semantic search
        print("DEBUG: Computing semantic similarity for metadata...")
        
        # Extract disease strings from metadata
        # Handle cases where 'diseases' might be missing or empty
        metadata_diseases = [m.get("diseases", "Unspecified") for m in all_metadatas]
        
        # Encode target and candidates
        target_emb = sbert_model.encode(target_disease, convert_to_tensor=True)
        candidate_embs = sbert_model.encode(metadata_diseases, convert_to_tensor=True)
        
        # Compute cosine similarity
        cosine_scores = util.cos_sim(target_emb, candidate_embs)[0]
        
        # Find top 2 matches
        k = 2
        # Ensure we don't ask for more than we have
        k = min(k, len(cosine_scores))
        
        top_k_result = cosine_scores.topk(k)
        top_indices = top_k_result.indices.tolist()
        top_scores = top_k_result.values.tolist()
        
        combined_context = ""
        
        # Save for inspection
        with open(RETRIEVED_FILE, "w", encoding="utf-8") as f:
            for rank, idx in enumerate(top_indices):
                score_val = top_scores[rank]
                chunk_id = all_ids[idx]
                disease_meta = metadata_diseases[idx]
                
                print(f"✅ Match #{rank+1}: '{target_disease}' ~ '{disease_meta}' (Score: {score_val:.4f})")
                print(f"   Chunk ID: {chunk_id}")
                
                # Retrieve the actual document content for this ID
                chunk_data = vector_store.get(ids=[chunk_id], include=["documents"])
                doc_content = chunk_data["documents"][0]
                
                # Append to context
                combined_context += f"\n\n=== SCHEMA OPTION {rank+1} ({disease_meta}) ===\n{doc_content}"
                
                # Write to file
                f.write(f"--- Matched Chunk #{rank+1} (ID: {chunk_id}) ---\n")
                f.write(f"Matched Disease Metadata: {disease_meta}\n")
                f.write(f"Similarity Score: {score_val:.4f}\n\n")
                f.write(doc_content)
                f.write("\n\n" + "="*50 + "\n\n")
        
        state["context"] = combined_context

    except Exception as e:
        print(f"❌ Metadata retrieval failed: {e}")
        state["context"] = ""

    state["retrieval_time"] = time.time() - start
    print(f"⏱️ Retrieval took {state['retrieval_time']:.2f} s")
    return state

class CloudflareLLM:
    def __init__(self, url):
        self.url = url.strip()

    def invoke(self, prompt):
        payload = {
            "model": "medgemma-27b-it",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 12188,
        }

        print(f"DEBUG: sending request to {self.url}")
        try:
            with requests.post(self.url, json=payload, stream=True, timeout=300) as response:
                if response.status_code != 200:
                    print(f"❌ HTTP Error: {response.status_code}")
                    print(f"Response text: {response.text[:500]}")
                    return f"[Error: HTTP {response.status_code}]"

                full_text = ""
                for line in response.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    
                    # Debug print first few lines to see what we're getting
                    # if len(full_text) < 100:
                    #    print(f"DEBUG LINE: {line}")

                    if line.startswith("data: "):
                        data = line[len("data: "):]

                        if data == "[DONE]":
                            break

                        try:
                            obj = json.loads(data)
                            if "choices" in obj and len(obj["choices"]) > 0:
                                delta = obj["choices"][0].get("delta", {}).get("content", "")
                                full_text += delta
                        except Exception as e:
                            print(f"JSON Parse Error on line: {line} -> {e}")
                            continue
                
                if not full_text:
                    print("⚠️ Warning: Received empty response from LLM")
                    
                return full_text
        except Exception as e:
            print(f"❌ STREAM PARSE ERROR: {e}")
            return f"[Error: {e}]"

def get_llm():
    tunnel_url = os.getenv(
        "MEDGEMMA_TUNNEL_URL",
        "https://sheer-installed-shorter-logistics.trycloudflare.com"
    )
    return CloudflareLLM(f"{tunnel_url}/v1/chat/completions")

# ==============================
# PROMPT + MODEL
# ==============================
prompt = get_discharge_summary_prompt(TRANSCRIPT_PATH, OPENEMR_PATH)
llm = get_llm()

def generate(state: RAGState):
    """
    Safe LLM call that bypasses LangChain message parsing.
    Eliminates the 'model_dump' error permanently.
    """


    start = time.time()

    # Diagnostic
    schema_len = len(state["context"])
    print(f"🧾 Original schema_context size: {schema_len/1000:.1f} KB")
    
    # Optional truncate
    if len(state["context"]) > 10000:
        print("⚠️ Schema context too large — compressing and truncating...")
        state["context"] = state["context"][:10000]

    # ------------------------------------------------------------------
    # NO LangChain chains; manual prompt + direct LLM invoke
    # ------------------------------------------------------------------
    try:
        final_prompt = prompt.format_prompt(
            schema_context=state["context"],
            question=state["question"]
        ).to_string()

        # Calculate input tokens
        try:
            import tiktoken
            encoding = tiktoken.encoding_for_model("gpt-4")
            input_tokens = len(encoding.encode(final_prompt))
            print(f"📊 Input tokens: {input_tokens:,}")
        except:
            input_tokens = len(final_prompt.split()) * 1.3  # Rough estimate
        print(f"📊 Input tokens (estimated): {int(input_tokens):,}")

        # DIRECT CALL (this avoids model_dump completely)
        response = llm.invoke(final_prompt)

        # Calculate output tokens
        try:
            output_tokens = len(encoding.encode(response if isinstance(response, str) else str(response)))
            print(f"📊 Output tokens: {output_tokens:,}")
            print(f"📊 Total tokens: {input_tokens + output_tokens:,}")
        except:
            output_tokens = len(str(response).split()) * 1.3  # Rough estimate
        print(f"📊 Output tokens (estimated): {int(output_tokens):,}")
        print(f"📊 Total tokens (estimated): {int(input_tokens + output_tokens):,}")

        # Endpoint may return raw string or ChatMessage
        if isinstance(response, str):
            state["answer"] = response
        elif hasattr(response, "content"):
            state["answer"] = response.content
        else:
            state["answer"] = str(response)

        # Store token counts
        state["input_tokens"] = int(input_tokens)
        state["output_tokens"] = int(output_tokens)
        state["total_tokens"] = int(input_tokens + output_tokens)
        print(f"DEBUG: Stored token counts in state: {state['total_tokens']}")

    except Exception as e:
        print(f"❌ LLM generation failed: {e}")
        state["answer"] = f"[Error during generation: {e}]"

    # Timing
    state["generation_time"] = time.time() - start
    print(f"⏱️ Generation took {state['generation_time']:.2f} s")
    
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

# Handle both success and error cases
if isinstance(result["answer"], str):
    generated_text = result["answer"]
elif hasattr(result["answer"], "content"):
    generated_text = result["answer"].content
else:
    generated_text = str(result["answer"])

# ==============================
# PHASE 3: EVALUATION
# ==============================
print("🔹 Starting Phase 3: Evaluation")

if not os.path.exists(REFERENCE_PATH):
    raise FileNotFoundError(f"⚠️ Reference file not found at {REFERENCE_PATH}.")

with open(REFERENCE_PATH, "r", encoding="utf-8") as ref_file:
    reference_text = ref_file.read().strip()

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
    f.write(f"Generation Time: {result.get('generation_time', 0):.2f} sec\n")
    f.write(f"\n=== Token Usage ===\n")
    f.write(f"Input Tokens: {result.get('input_tokens', 'N/A')}\n")
    f.write(f"Output Tokens: {result.get('output_tokens', 'N/A')}\n")
    f.write(f"Total Tokens: {result.get('total_tokens', 'N/A')}\n\n")
    f.write("--- Generated Summary ---\n")
    f.write(generated_text)

print(f"✅ Evaluation completed — saved to {RESULT_FILE}")

# cd rag_models\RAG_To_See_MedGemma_Performance