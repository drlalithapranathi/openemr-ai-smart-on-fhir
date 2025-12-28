"""
Modal-based SOAP summarization using MedGemma 4B-IT with RAG
Uses disease-specific schemas from ChromaDB to guide extraction
"""

import modal
import json
import time
from typing import Dict, Any
from pydantic import BaseModel

# ============================================================================
# Modal App Configuration
# ============================================================================

app = modal.App("soap-summarization-medgemma-4b-test")

# Persistent volume for vector database
vectordb_volume = modal.Volume.from_name("medical-vectordb")

# Model configuration
MODEL_NAME = "google/medgemma-4b-it"
CHROMA_PATH = "/vectordb/chroma_schema_improved"

# ============================================================================
# Modal Image
# ============================================================================

summarizer_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi",
        "langchain>=0.1.0",
        "langchain-community>=0.0.20",
        "langchain-huggingface>=0.0.1",
        "langchain-chroma>=0.1.0",
        "sentence-transformers>=2.2.2",
        "chromadb>=0.4.22",
        "transformers>=4.45.0",
        "torch>=2.1.0",
        "accelerate>=0.25.0",
        "bitsandbytes>=0.41.0",
        "huggingface-hub>=0.20.0",
    )
)

# ============================================================================
# Request Model for HTTP endpoint
# ============================================================================

class SummarizeRequest(BaseModel):
    transcript_text: str
    openemr_text: str = ""
    patient_name: str = "Patient"


# ============================================================================
# Helper: Convert JSON schema to prose instructions
# ============================================================================

def schema_to_prose(schema_json: str, disease: str) -> str:
    """
    Convert JSON schema to natural language extraction guide.
    This prevents MedGemma from mimicking JSON format in output.
    """
    try:
        schema = json.loads(schema_json)
    except:
        return f"For {disease}, extract standard clinical information."

    parts = [f"For {disease}, look for the following relevant information:"]

    # Extract lab analytes
    labs = schema.get("objective", {}).get("labs", {})
    if labs:
        panels = labs.get("panels", [])
        analytes = []
        for panel in panels:
            analytes.extend(panel.get("analytes", []))
        if analytes:
            analyte_list = ", ".join(analytes[:20])
            if len(analytes) > 20:
                analyte_list += f" (and {len(analytes) - 20} more)"
            parts.append(f"- Relevant lab tests: {analyte_list}")

    # Extract imaging types
    imaging = schema.get("objective", {}).get("imaging", [])
    study_types = [img.get("study_type") for img in imaging if img.get("study_type")]
    if study_types:
        parts.append(f"- Relevant imaging: {', '.join(study_types)}")

    # Vital signs
    vitals = schema.get("objective", {}).get("vital_signs", {})
    if vitals:
        vital_keys = [k.upper() for k in vitals.keys()]
        if vital_keys:
            parts.append(f"- Vital signs to note: {', '.join(vital_keys)}")

    # Microbiology
    micro = labs.get("microbiology", [])
    if micro:
        parts.append("- Check for any microbiology/culture results")

    return "\n".join(parts)


# ============================================================================
# Helper: Format FHIR data as readable text
# ============================================================================

def format_fhir_as_text(fhir_json: str) -> str:
    """
    Convert FHIR JSON data to readable clinical text for the prompt.
    """
    if not fhir_json or fhir_json.strip() == "":
        return "No EHR data available."

    try:
        data = json.loads(fhir_json)
    except:
        return "No EHR data available."

    sections = []

    # Patient info
    patient = data.get("patient", {})
    if patient:
        sections.append(f"Patient: {patient.get('name', 'Unknown')}, DOB: {patient.get('dob', 'Unknown')}, Gender: {patient.get('gender', 'Unknown')}")

    # Vitals
    vitals = data.get("vitals", [])
    if vitals:
        vital_strs = [f"{v.get('type', '')}: {v.get('value', '')}" for v in vitals[:10]]
        sections.append("VITAL SIGNS:\n" + "\n".join(vital_strs))

    # Labs
    labs = data.get("labs", [])
    if labs:
        lab_strs = []
        for lab in labs[:15]:
            lab_str = f"{lab.get('test', '')}: {lab.get('value', '')}"
            if lab.get('refRange'):
                lab_str += f" (ref: {lab.get('refRange')})"
            if lab.get('interpretation'):
                lab_str += f" [{lab.get('interpretation')}]"
            lab_strs.append(lab_str)
        sections.append("LABORATORY RESULTS:\n" + "\n".join(lab_strs))

    # Conditions
    conditions = data.get("conditions", [])
    if conditions:
        cond_strs = [f"- {c.get('name', '')} ({c.get('status', '')})" for c in conditions]
        sections.append("ACTIVE CONDITIONS:\n" + "\n".join(cond_strs))

    # Allergies
    allergies = data.get("allergies", [])
    if allergies:
        allergy_strs = [f"- {a.get('substance', '')}: {a.get('reaction', '')} ({a.get('severity', '')})" for a in allergies]
        sections.append("ALLERGIES:\n" + "\n".join(allergy_strs))

    # Medications
    medications = data.get("medications", [])
    if medications:
        med_strs = [f"- {m.get('name', '')}: {m.get('dosage', '')}" for m in medications]
        sections.append("CURRENT MEDICATIONS:\n" + "\n".join(med_strs))

    # Imaging
    imaging = data.get("imaging", [])
    if imaging:
        img_strs = [f"- {i.get('type', '')}: {i.get('conclusion', '')}" for i in imaging]
        sections.append("IMAGING RESULTS:\n" + "\n".join(img_strs))

    if not sections:
        return "No EHR data available."

    return "\n\n".join(sections)


# ============================================================================
# SOAP Summarizer Class
# ============================================================================

@app.cls(
    image=summarizer_image,
    gpu="A10G",
    timeout=3600,
    volumes={"/vectordb": vectordb_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
class SOAPSummarizer:
    """
    RAG-based medical summarizer using MedGemma 4B-IT.
    Models are loaded once in @modal.enter() and reused across all calls.
    """

    @modal.enter()
    def load_models(self):
        """Load all models once when container starts."""
        import torch
        from transformers import pipeline
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_chroma import Chroma
        from sentence_transformers import SentenceTransformer

        print("🔄 Loading models (one-time initialization)...")
        print(f"   Model: {MODEL_NAME}")

        # Load MedGemma 4B-IT pipeline
        print("  → Loading MedGemma 4B-IT pipeline...")
        start_load = time.time()

        self.pipe = pipeline(
            "text-generation",
            model=MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        load_time = time.time() - start_load
        print(f"  → MedGemma loaded in {load_time:.2f}s")

        # Load BioBERT embeddings for ChromaDB
        print("  → Loading BioBERT embeddings...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
        )

        # Load vector store
        print(f"  → Loading Vector Store from: {CHROMA_PATH}")
        self.vector_store = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=self.embeddings
        )

        # Pre-fetch collection data
        print("  → Pre-fetching collection data...")
        collection_data = self.vector_store.get(include=["metadatas", "documents"])
        self.all_metadatas = collection_data["metadatas"]
        self.all_docs = collection_data["documents"]
        self.metadata_diseases = [m.get("diseases", "Unspecified") for m in self.all_metadatas]

        # Load SBERT for semantic matching
        print("  → Loading SBERT model...")
        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Pre-encode all disease metadata
        print("  → Pre-encoding disease embeddings...")
        self.candidate_embs = self.sbert_model.encode(self.metadata_diseases, convert_to_tensor=True)

        print("✅ All models loaded successfully!")

    def _generate_text(self, messages: list, max_new_tokens: int = 2048, temperature: float = 0.3) -> str:
        """Generate text using the pipeline."""
        output = self.pipe(messages, max_new_tokens=max_new_tokens, temperature=temperature)
        return output[0]["generated_text"][-1]["content"]

    @modal.method()
    def generate_summary(
            self,
            transcript_text: str,
            openemr_text: str = "",
            patient_name: str = "Patient",
    ) -> Dict[str, Any]:
        """
        Generate medical summary from transcript + FHIR data using RAG.
        """
        from sentence_transformers import util

        print(f"\n{'='*60}")
        print(f"🔹 Generating summary for: {patient_name}")
        print(f"🔹 Transcript length: {len(transcript_text)} chars")
        print(f"🔹 FHIR data length: {len(openemr_text)} chars")
        print(f"{'='*60}")
        start_total = time.time()

        # ==============================
        # 1. EXTRACT DISEASE USING MEDGEMMA
        # ==============================
        print("🔹 Extracting disease from transcript...")
        start_retrieval = time.time()

        disease_messages = [
            {
                "role": "system",
                "content": "You are a medical expert. Identify the primary medical condition from clinical conversations. Respond with ONLY the disease name, nothing else."
            },
            {
                "role": "user",
                "content": f"""Read this transcript and identify the PRIMARY medical condition being discussed.

Return ONLY the disease name (e.g., "COPD", "Diabetes", "Hypertension", "Asthma", "Heart Failure").
If no specific disease is mentioned, return "General".

Transcript:
{transcript_text[:2000]}

Primary Disease:"""
            }
        ]

        detected_disease = self._generate_text(disease_messages, max_new_tokens=20).strip()

        # Clean up
        if not detected_disease:
            detected_disease = "General"
        detected_disease = detected_disease.split('\n')[0].split(',')[0].strip()

        print(f"✅ Detected Disease: {detected_disease}")

        # ==============================
        # 2. RETRIEVE SCHEMAS FROM VECTOR DB
        # ==============================
        print("🔹 Retrieving relevant schemas from vector DB...")

        target_emb = self.sbert_model.encode(detected_disease, convert_to_tensor=True)
        cosine_scores = util.cos_sim(target_emb, self.candidate_embs)[0]

        # Get top 2 schemas
        k = min(2, len(cosine_scores))
        top_k_result = cosine_scores.topk(k)
        top_indices = top_k_result.indices.tolist()

        # Convert schemas to PROSE (not JSON!)
        schema_guidance = ""
        for rank, idx in enumerate(top_indices):
            doc_content = self.all_docs[idx]
            disease_meta = self.metadata_diseases[idx]
            prose_guide = schema_to_prose(doc_content, disease_meta)
            schema_guidance += f"\n{prose_guide}\n"

        retrieval_time = time.time() - start_retrieval
        print(f"⏱️ Disease extraction + retrieval: {retrieval_time:.2f}s")

        # ==============================
        # 3. FORMAT FHIR DATA AS TEXT
        # ==============================
        ehr_text = format_fhir_as_text(openemr_text)
        print(f"🔹 EHR data formatted: {len(ehr_text)} chars")

        # ==============================
        # 4. GENERATE SUMMARY WITH MEDGEMMA
        # ==============================
        print("🔹 Generating medical summary with MedGemma 4B-IT...")
        start_gen = time.time()

        summary_messages = [
            {
                "role": "system",
                "content": "You are an expert medical scribe specialized in clinical documentation. Generate comprehensive medical summaries in narrative prose format."
            },
            {
                "role": "user",
                "content": f"""Generate a comprehensive medical summary from the following data:

### TRANSCRIPT (Doctor-patient conversation):
{transcript_text}

### ELECTRONIC HEALTH RECORD DATA:
{ehr_text}

### SCHEMA GUIDE (Reference for relevant sections):
{schema_guidance}

### OUTPUT FORMAT REQUIREMENTS
- Generate a NARRATIVE TEXT document, NOT JSON or structured data
- Use clear section headers (Patient Information, Chief Complaint, History of Present Illness, Past Medical History, Medications, Allergies, Review of Systems, Physical Exam, Labs, Imaging, Assessment, Plan)
- Write in complete sentences and paragraphs
- Use professional medical documentation prose style

### INSTRUCTIONS
1. Extract relevant information from the TRANSCRIPT and EHR DATA
2. Write in narrative prose with proper paragraphs
3. If information for a section is missing, write "No information available."
4. Do NOT output JSON, XML, bullet points, or any structured data format
5. Do NOT hallucinate or invent information not present in the inputs

Generate the medical summary now:

Patient Information:
{patient_name}

Chief Complaint:"""
            }
        ]

        try:
            generated_text = self._generate_text(summary_messages, max_new_tokens=2048)

            # Prepend the patient info and chief complaint header
            generated_text = f"Patient Information:\n{patient_name}\n\nChief Complaint:\n" + generated_text

        except Exception as e:
            print(f"❌ MedGemma generation failed: {e}")
            generated_text = f"Error generating summary: {str(e)}"

        generation_time = time.time() - start_gen
        total_time = time.time() - start_total

        print(f"⏱️ Generation: {generation_time:.2f}s | Total: {total_time:.2f}s")
        print(f"📝 Output preview: {generated_text[:200]}...")

        return {
            "success": True,
            "soap_note": generated_text,
            "summary": generated_text,
            "detected_disease": detected_disease,
            "patient_name": patient_name,
            "retrieval_time": retrieval_time,
            "generation_time": generation_time,
            "total_time": total_time,
            "model": MODEL_NAME,
        }

    @modal.web_endpoint(method="POST")
    def summarize(self, request: SummarizeRequest) -> Dict[str, Any]:
        """HTTP endpoint for SOAP note generation."""
        return self.generate_summary(
            transcript_text=request.transcript_text,
            openemr_text=request.openemr_text,
            patient_name=request.patient_name,
        )
