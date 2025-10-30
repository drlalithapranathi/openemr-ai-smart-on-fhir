import os
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document


def setup_vector_store(data_path: str, chroma_path: str):
    """
    Loads or builds a Chroma vector store safely and efficiently.
    Automatically detects if an existing database is available.
    """

    print("🔹 Starting Phase 1: Indexing")
    start = time.time()

    # --- 1️⃣ Check if Chroma database already exists ---
    if os.path.exists(chroma_path) and len(os.listdir(chroma_path)) > 0:
        print("⚙️ Existing ChromaDB found — loading only (skipping re-indexing).")
        embeddings = HuggingFaceEmbeddings(
            model_name="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
        )
        vector_store = Chroma(
            persist_directory=chroma_path,
            embedding_function=embeddings
        )
        print("✅ ChromaDB loaded successfully.\n")
        return vector_store

    # --- 2️⃣ If no DB exists, start fresh indexing ---
    print("🆕 No existing ChromaDB found — creating new one...")
    os.makedirs(chroma_path, exist_ok=True)

    # --- 3️⃣ Initialize embedding model ---
    embeddings = HuggingFaceEmbeddings(
        model_name="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
    )

    # --- 4️⃣ Initialize Chroma (empty store for now) ---
    vector_store = Chroma(
        persist_directory=chroma_path,
        embedding_function=embeddings
    )

    # --- 5️⃣ Splitter setup ---
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]
    )

    # --- 6️⃣ Stream file and add to Chroma in batches ---
    batch_size = 20000  # lines per batch
    total_chunks = 0
    buffer = []

    print(f"📂 Reading from: {data_path}")
    print(f"💾 Saving ChromaDB to: {chroma_path}\n")

    with open(data_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            buffer.append(line.strip())

            # Every 20k lines → process a batch
            if (i + 1) % batch_size == 0:
                text_block = "\n".join(buffer)
                buffer = []

                docs = splitter.create_documents([text_block])
                total_chunks += len(docs)

                print(f"🧩 Indexed lines {i+1-batch_size}-{i+1}: {len(docs)} chunks")
                vector_store.add_documents(docs)

        # Remaining lines (if any)
        if buffer:
            text_block = "\n".join(buffer)
            docs = splitter.create_documents([text_block])
            total_chunks += len(docs)
            vector_store.add_documents(docs)

    # --- 7️⃣ Persist database ---
    print("\n💾 Persisting Chroma vector store to disk...")
    vector_store.persist()

    print(f"✅ Indexed {total_chunks:,} chunks in {time.time() - start:.2f} seconds.")
    print(f"✅ Vector store saved successfully at: {chroma_path}\n")

    return vector_store
