import os
import time
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

# ==============================
# CONFIGURATION
# ==============================
load_dotenv()

DATA_PATH = "data/MIMIC_NOTE.txt"
CHROMA_PATH = "vectorDB/chroma/"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
RESULT_FILE = os.path.join(RESULTS_DIR, "evaluation.txt")


# ==============================
# PHASE 1: INDEXING
# ==============================
print("🔹 Starting Phase 1: Indexing")
start_time = time.time()

# Step 1a. Load Data
with open(DATA_PATH, "r", encoding="utf-8") as f:
    text_data = f.read()
# print(text_data)

# Step 1b. Split Text into Chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1600, chunk_overlap=160)
docs = splitter.create_documents([text_data])
# print(docs[1])

# Step 1c. Embedding Model
embeddings = HuggingFaceEmbeddings(model_name="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")
# embeddings = HuggingFaceEmbeddings(model_name="emilyalsentzer/Bio_ClinicalBERT")


# Step 1d. Check if Chroma already exists
if os.path.exists(CHROMA_PATH) and len(os.listdir(CHROMA_PATH)) > 0:
    print("⚙️ Existing ChromaDB found — loading from disk...")
    vector_store = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )
else:
    print("🆕 No existing ChromaDB found — creating new vector store...")
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    print("💾 New ChromaDB created and saved automatically.")

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 7})
# print("✅ Phase 1 completed — Vector DB ready\n")

# Uncomment for debugging
print(f"Total chunks: {len(docs)}")
sample = vector_store._collection.count()
print(f"Vector store now holds {sample} documents.")

end_time = time.time()
print(f"⏱️ Phase 1 completed in {end_time - start_time:.2f} seconds\n")

# ==============================
# PHASE 2: RETRIEVAL + GENERATION
# ==============================
print("🔹 Starting Phase 2: Retrieval + Generation")
# start_time = time.time()

prompt = PromptTemplate(
    template=(
        "You are a clinical AI model highly specialized in writing complete, factual, and well-structured "
        "hospital discharge summaries for the condition specified by the user. "
        "Use only the information provided in the MIMIC-IV notes as your factual source. "
        "Do not create placeholders or generic templates. If a piece of information is missing, clearly write 'Information not available.'\n\n"

        "Your goal is to produce a **fully written discharge summary**, not a template. "
        "Write naturally, with clear paragraph-style sentences under each heading. "
        "Each section must be filled with content derived or synthesized from the provided notes.\n\n"

        "Follow this structure strictly while ensuring the output reads as a finalized discharge document:\n\n"

        "1. **Patient Information:**\n"
        "   - Name: [From context or 'Information not available']\n"
        "   - Unit No: [From context or 'Information not available']\n"
        "   - Admission Date: [From context or 'Information not available']\n"
        "   - Discharge Date: [From context or 'Information not available']\n"
        "   - Date of Birth: [From context or 'Information not available']\n"
        "   - Sex: [From context or 'Information not available']\n"
        "   - Service: Medicine\n"
        "   - Allergies: [From context or 'Information not available']\n"
        "   - Attending Physician: [From context or 'Information not available']\n\n"

        "2. **Chief Complaint:**\n"
        "   Provide a clear one-sentence statement summarizing the primary reason for admission related to {question}.\n\n"

        "3. **Major Procedures:**\n"
        "   List all procedures or write 'None performed during this admission.'\n\n"

        "4. **History of Present Illness (HPI):**\n"
        "   Write a cohesive paragraph describing the onset, duration, and clinical course of {question}, including key symptoms, comorbidities, "
        "and relevant background from the context.\n\n"

        "5. **Past Medical, Social, and Family History:**\n"
        "   Summarize any chronic conditions, medications, habits, and relevant family history mentioned in the notes, but keep all 3 of these in separate paragraph.\n\n"

        "6. **Physical Examination:**\n"
        "   - On Admission: summarize the key findings related to {question}.\n"
        "   - On Discharge: describe the patient's clinical status at discharge.\n\n"

        "7. **Pertinent Results:**\n"
        "   Summarize essential laboratory and imaging results that relate to {question}.\n"
        "   Include both admission and discharge data if available.\n\n"

        "8. **Hospital Course:**\n"
        "   Write a detailed narrative describing diagnostic evaluation, treatments given, progress, and outcome during the stay. "
        "Focus on the main condition ({question}) and related management steps.\n\n"

        "9. **Chronic and Transitional Issues:**\n"
        "   Describe ongoing conditions managed during hospitalization and care plans after discharge.\n\n"

        "10. **Medications:**\n"
        "   - On Admission: list known pre-hospital medications.\n"
        "   - On Discharge: list medications with dosage, frequency, and purpose.\n\n"

        "11. **Discharge Details:**\n"
        "   - Discharge Disposition (e.g., Home, Rehabilitation, Skilled Nursing Facility)\n"
        "   - Primary Diagnosis: {question}\n"
        "   - Secondary Diagnoses: [From context or 'Information not available']\n"
        "   - Condition at Discharge: [Describe clearly, not as a placeholder]\n\n"

        "12. **Patient Instructions:**\n"
        "   Provide actual discharge instructions in natural, patient-friendly language "
        "about medications, follow-up, diet, and activity.\n\n"

        "13. **Follow-up Plan:**\n"
        "   Mention the next steps, including follow-up appointments, tests, or consultations.\n\n"
        
        "---\n\n"
        "**Context (MIMIC-IV Notes):**\n{context}\n\n"
        "Return only the final formatted discharge summary as normal text. Do not include markdown code fences or language tags like ```json or ```."

        "Now, write one complete discharge summary for this hospitalization related to {question}, using the above structure. "
        "Do not provide examples or multiple summaries. Write only the final, filled summary."
    ),
    input_variables=["context", "question"],
)



# Using LM Studio MedGemma 
llm = ChatOpenAI(
    model_name="medgemma-27b-it",
    # base_url="http://192.168.1.128:1234/v1",
    base_url="https://lancinate-persuasive-paxton.ngrok-free.dev/v1",
    api_key="not-needed", # The API key can be anything
    temperature=0.5
    # max_tokens=2048
)
# result = llm.invoke("What is the capital of France?")
# print(result)


# Define state
class RAGState(dict):
    question: str
    context: str
    answer: str

# Define nodes
def retrieve(state: RAGState):
    start = time.time()
    docs = retriever.invoke(state["question"])
    context = "\n\n".join(doc.page_content for doc in docs)
    state["context"] = context
    end = time.time()
    state["retrieval_time"] = end - start   # ⏱️ save retrieval duration
    print(f"⏱️ Retrieval took {state['retrieval_time']:.2f} seconds")
    return state

def generate(state: RAGState):
    start = time.time()
    # prompt = prompt
    chain = prompt | llm
    state["answer"] = chain.invoke({
        "context": state["context"],
        "question": state["question"]
    })
    end = time.time()
    state["generation_time"] = end - start   # ⏱️ save generation duration
    print(f"⏱️ Generation took {state['generation_time']:.2f} seconds")
    return state

# Build the graph
graph = StateGraph(RAGState)
graph.add_node("retrieve", retrieve)
graph.add_node("generate", generate)

graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)

app = graph.compile()

# Run it
result = app.invoke({"question": "Generate a discharge summary for a patient admitted with COPD"})
# print(result["answer"])

# end_time = time.time()
print(f"⏱️ Phase 2 completed — Summary generated\n")


# ==============================
# PHASE 3: EVALUATION
# ==============================
print("🔹 Starting Phase 3: Evaluation")
start_time = time.time()

REFERENCE_PATH = os.path.join("data", "reference.txt")

# Step 3a. Load reference text
if not os.path.exists(REFERENCE_PATH):
    raise FileNotFoundError(
        f"⚠️ Reference file not found at {REFERENCE_PATH}. "
        "Please add 'reference.txt' in the data folder."
    )

with open(REFERENCE_PATH, "r", encoding="utf-8") as ref_file:
    reference_text = ref_file.read().strip()

# Uncomment to verify reference content
# print("\n--- Reference Text Preview ---\n")
# print(reference_text[:500], "\n")

# Step 3b. Compute BLEU
# -------------------------------------------------------
# `generated_text` holds the LLM output from Phase 2.
# `reference_text` is the true discharge summary.
# -------------------------------------------------------
generated_text = result["answer"].content

bleu_score = sentence_bleu(
    [reference_text.split()],
    generated_text.split()
)

# Step 3c. Compute ROUGE-L
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
rouge_l = scorer.score(reference_text, generated_text)["rougeL"].fmeasure

# Step 3d. Save evaluation results
with open(RESULT_FILE, "w", encoding="utf-8") as f:
    f.write("=== Evaluation Metrics ===\n")
    f.write(f"User Query: {result.get('question', 'N/A')}\n\n")
    f.write(f"BLEU Score: {bleu_score:.4f}\n")
    f.write(f"ROUGE-L Score: {rouge_l:.4f}\n")
    # f.write("\n--- Reference Text ---\n")
    # f.write(reference_text[:1000] + "\n...\n\n")
    f.write("--- Generated Summary ---\n")
    f.write(generated_text)

end_time = time.time()
print(f"⏱️ Phase 3 completed in {end_time - start_time:.2f} seconds — Results saved to {RESULT_FILE}\n")

# Uncomment below for quick terminal printout
# print("=== Evaluation Metrics ===")
# print(f"BLEU Score: {bleu_score:.4f}")
# print(f"ROUGE-L Score: {rouge_l:.4f}")
