from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import os

def _read(path: str, fallback: str) -> str:

    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()

            return text if text else fallback
    except FileNotFoundError:

        return fallback


def get_discharge_summary_prompt(audio_path: str, openemr_path: str):
        """
        Minimal, strict, structure-driven discharge summary prompt.
        Schema Context defines all sections/subsections. 
        Transcript and OpenEMR define all content.
        """

        audio = _read(audio_path, "No live conversation available.")
        openemr = _read(openemr_path, "No OpenEMR extract available.")

        return PromptTemplate(
            template=(
                "You are an expert medical scribe. Your task is to write a comprehensive Hospital Discharge Summary for the patient described in the provided data.\n\n"

                "### INPUT DATA\n"
                "1. **TRANSCRIPT**: Real-time dialogue from the current visit. Use this for the Chief Complaint, History of Present Illness (HPI), Review of Systems (ROS), and current Plan.\n"
                "2. **OPENEMR EXTRACT**: The patient's electronic health record. Use this for Patient Demographics, Past Medical History, Surgical History, Medications, and Allergies.\n"
                "3. **SCHEMA GUIDE**: A JSON structure that outlines the required sections and subsections for this specific condition.\n\n"

                "### INSTRUCTIONS\n"
                "- **Structure**: Organize your summary exactly according to the sections defined in the **SCHEMA GUIDE**. Use the keys in the schema as your section headings (e.g., 'History of Present Illness', 'Hospital Course').\n"
                "- **Content**: Fill each section with relevant information from the **TRANSCRIPT** and **OPENEMR EXTRACT**.\n"
                "- **Missing Info**: If a section in the schema requires information that is NOT in the transcript or OpenEMR, write 'Information not available'.\n"
                "- **Style**: Write in full, professional sentences for the narrative sections (HPI, Hospital Course). Use bullet points for lists (Meds, Vitals, History).\n"
                "- **Conflict Resolution**: If the Transcript contradicts the OpenEMR (e.g., patient says they stopped a med), trust the **TRANSCRIPT** for the current status.\n\n"

                "### CRITICAL: STRICT SCHEMA ADHERENCE (DO NOT SUMMARIZE)\n"
                "1. **COPY-PASTE REQUIREMENT**: You must output the EXACT structure from the SCHEMA GUIDE. Do not change headers. Do not summarize.\n"
                "2. **FORBIDDEN HEADERS**: Do NOT use generic headers like 'Pertinent Results', 'Brief Hospital Course', or 'Relevant Labs'. Use ONLY the keys provided in the Schema.\n"
                "3. **NEGATIVE CONSTRAINTS**: \n"
                "   - NEVER omit a section because it is empty.\n"
                "   - NEVER summarize a list of labs into a sentence.\n"
                "   - NEVER say 'Labs are pending' as a summary. List each lab and write 'Pending' or 'Not available' next to it.\n"
                "4. **ONE-SHOT EXAMPLE (FOLLOW THIS PATTERN)**: \n"
                "   **Input Schema**:\n"
                "   {{\n"
                "     \"labs\": [{{\"panel_name\": \"CBC\", \"analytes\": [\"WBC\", \"RBC\"]}}]\n"
                "   }}\n"
                "   **Correct Output**:\n"
                "   Labs:\n"
                "     - CBC:\n"
                "       - WBC: Not available\n"
                "       - RBC: 4.5\n"
                "   **Incorrect Output (DO NOT DO THIS)**:\n"
                "   Pertinent Results: CBC was ordered but results are not available.\n"
                "5. **ZERO HALLUCINATION**: If data is missing, write 'Not available'. Do NOT make up values. Do NOT skip the item.\n\n"

                "### REQUIRED SECTIONS (Based on Schema)\n"
                "Please ensure you include **Patient Information** at the very top (Name, DOB, MRN, Admission/Discharge Dates), followed by the SOAP sections as outlined in the Schema Guide.\n\n"

                "--- START OF DATA ---\n\n"
                "**TRANSCRIPT**:\n{audio_transcript}\n\n"
                "**OPENEMR EXTRACT**:\n{openemr_extract}\n\n"
                "**SCHEMA GUIDE**:\n{schema_context}\n\n"
                "--- END OF DATA ---\n\n"

                "**OUTPUT**:\n"
                "Please generate the full discharge summary below. Do not output JSON. Output clean, formatted text.\n"
            ),
            input_variables=["schema_context", "question"],
            partial_variables={
                "audio_transcript": audio,
                "openemr_extract": openemr,
            },
        )




# def get_llm():
#     """
#     Initializes and returns the MedGemma LLM connection for generating
#     high-fidelity clinical discharge summaries.
#     """
#     llm = ChatOpenAI(
#         model_name="medgemma-27b-it",
#         base_url="https://lancinate-persuasive-paxton.ngrok-free.dev/v1",
#         api_key="not-needed",
#         temperature=0.35,
#         top_p=0.95,
#         frequency_penalty=0.3,
#         presence_penalty=0.2,
#         request_timeout=180,
#     )
#     return llm

# def get_llm():
#     """Initializes and returns the MedGemma LLM connection."""
#     llm = ChatOpenAI(
#         model_name="medgemma-27b-it",
#         # model_name="medgemma-4b-it",
#         # base_url="http://192.168.1.128:1234/v1",
#         # base_url="https://lancinate-persuasive-paxton.ngrok-free.dev/v1",
#         base_url="https://berkeley-computers-karaoke-provides.trycloudflare.com/v1",
#         # base_url="https://sie-consult-ranch-jail.trycloudflare.com/v1",
#         api_key="not-needed",
#         temperature=0.3
#     )
#     return llm

# def get_llm():
#     """Initializes and returns the MedGemma LLM connection."""
#     tunnel_url = os.getenv(
#         "MEDGEMMA_TUNNEL_URL",
#         "https://sheer-installed-shorter-logistics.trycloudflare.com  # fallback
#     )
#     llm = ChatOpenAI(
#         model_name="medgemma-27b-it",
#         base_url=f"{tunnel_url}/v1",
#         api_key="not-needed",
#         temperature=0.3,
#         default_headers={
#             "User-Agent": "OpenAI-Python-Client",
#             "Accept": "application/json"
#         }
#     )
#     return llm
