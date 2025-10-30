import os, time, re, shutil
import random
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

chroma_path = "vectorDB/chroma_test/"
data_path = "data/Top_100.txt"

def setup_test_vector_store(data_path: str, chroma_path: str):
    """
    Builds or loads a section-aware Chroma vector store safely.
    Re-indexes only if no existing database is found.
    """
    print("🔹 Starting Section-Aware Indexing (Test Mode)")
    start = time.time()

    # ✅ 1. Check if DB already exists
    if os.path.exists(chroma_path) and len(os.listdir(chroma_path)) > 0:
        print("⚙️ Existing ChromaDB found — loading (skipping re-indexing).")
        embeddings = HuggingFaceEmbeddings(
            model_name="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
        )
        db = Chroma(persist_directory=chroma_path, embedding_function=embeddings)
        print(f"✅ Loaded existing ChromaDB from {chroma_path}\n")
        return db

    # 🆕 2. If not found, build new vector store
    print("🆕 No existing ChromaDB found — creating new one...")

    # Clean up (if needed)
    if os.path.exists(chroma_path):
        try:
            shutil.rmtree(chroma_path)
            print(f"🧹 Removed old directory: {chroma_path}")
        except PermissionError:
            temp_path = chroma_path + "_old"
            os.rename(chroma_path, temp_path)
            shutil.rmtree(temp_path, ignore_errors=True)

    # 3️⃣ Initialize embeddings + Chroma
    embeddings = HuggingFaceEmbeddings(
        model_name="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
    )
    db = Chroma(persist_directory=chroma_path, embedding_function=embeddings)

    # 4️⃣ Define section headers
    pattern = re.compile(
        r"(?=^(?:Name:|Admission Date:|Discharge Date:|Chief Complaint:|"
        r"History of Present Illness:|Physical Exam:|Physical Examination:|"
        r"Hospital Course:|Pertinent Results:|Labs?:|Imaging:|Medications?:|"
        r"Social History:|Family History:|Allergies:|Discharge Condition:|"
        r"Discharge Diagnosis:|Discharge Instructions:|Followup Instructions:))",
        re.MULTILINE
    )

    # 5️⃣ Read + split text
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 🧹 Clean duplicate headers like "Discharge Discharge"
    text = re.sub(r'\b(Discharge|Admission|Medications)\s+\1\b', r'\1', text)
    text = re.sub(r'\n{2,}', '\n', text).strip()

    # 6️⃣ Split into sections and chunks
    sections = re.split(pattern, text)
    sections = [s.strip() for s in sections if s.strip()]

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = splitter.create_documents(sections)
    db.add_documents(docs)

    # 7️⃣ Summary
    print(f"📄 Loaded {len(text):,} characters from {data_path}")
    print(f"🧩 Found {len(sections)} top-level sections")
    print(f"🧠 Created {len(docs)} final section-aware chunks")
    print(f"✅ ChromaDB saved at: {chroma_path}")
    print(f"⏱️ Total time: {time.time() - start:.2f} sec\n")

    return db


# 🔍 Verify chunking visually
data_path = "data/Top_100.txt"
chroma_path = "vectorDB/chroma_test/"

db = setup_test_vector_store(data_path, chroma_path)
docs = db.get()['documents']
print(f"✅ Total Chunks: {len(docs)}")

sample_indices = random.sample(range(len(docs)), 10)
for i in sample_indices:
    print(f"\n--- Random Chunk {i+1} ---\n{docs[i][:500]}\n{'='*80}")

