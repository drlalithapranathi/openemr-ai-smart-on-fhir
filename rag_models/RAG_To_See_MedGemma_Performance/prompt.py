# prompt.py
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# def get_discharge_summary_prompt():
#     """Creates and returns the discharge summary prompt template."""
#     return PromptTemplate(
#         template=(
#         "You are a clinical AI model highly specialized in writing complete, factual, and well-structured "
#         "hospital discharge summaries for the condition specified by the user. "
#         "Use only the information provided in the MIMIC-IV notes as your factual source. "
#         "Do not invent patient names or data, but you may infer clinical flow and continuity from context. "
#         "If specific fields are missing, briefly summarize relevant information instead of leaving placeholders.\n\n"
    

#         "Your goal is to produce a **fully written discharge summary**, not a template. "
#         "Write naturally, with clear paragraph-style sentences under each heading. "
#         "Each section must be filled with content derived or synthesized from the provided notes.\n\n"

#         "Follow this structure strictly while ensuring the output reads as a finalized discharge document:\n\n"

#         "1. **Patient Information:**\n"
#         "   - Name: [From context or 'Information not available']\n"
#         "   - Unit No: [From context or 'Information not available']\n"
#         "   - Admission Date: [From context or 'Information not available']\n"
#         "   - Discharge Date: [From context or 'Information not available']\n"
#         "   - Date of Birth: [From context or 'Information not available']\n"
#         "   - Sex: [From context or 'Information not available']\n"
#         "   - Service: Medicine\n"
#         "   - Allergies: [From context or 'Information not available']\n"
#         "   - Attending Physician: [From context or 'Information not available']\n\n"

#         "2. **Chief Complaint:**\n"
#         "   Provide a clear one-sentence statement summarizing the primary reason for admission related to {question}.\n\n"

#         "3. **Major Procedures:**\n"
#         "   List all procedures or write 'None performed during this admission.'\n\n"

#         "4. **History of Present Illness (HPI):**\n"
#         "   Write a cohesive paragraph describing the onset, duration, and clinical course of {question}, including key symptoms, comorbidities, "
#         "and relevant background from the context.\n\n"

#         "5. **Past Medical, Social, and Family History:**\n"
#         "   Summarize any chronic conditions, medications, habits, and relevant family history mentioned in the notes, but keep all 3 of these in separate paragraph.\n\n"

#         "6. **Physical Examination:**\n"
#         "   - On Admission: summarize the key findings related to {question}.\n"
#         "   - On Discharge: describe the patient's clinical status at discharge.\n\n"

#         "7. **Pertinent Results:**\n"
#         "   Summarize essential laboratory and imaging results that relate to {question}.\n"
#         "   Include both admission and discharge data if available.\n\n"

#         "8. **Hospital Course:**\n"
#         "   Write a detailed narrative describing diagnostic evaluation, treatments given, progress, and outcome during the stay. "
#         "Focus on the main condition ({question}) and related management steps.\n\n"

#         "9. **Chronic and Transitional Issues:**\n"
#         "   Describe ongoing conditions managed during hospitalization and care plans after discharge.\n\n"

#         "10. **Medications:**\n"
#         "   - On Admission: list known pre-hospital medications.\n"
#         "   - On Discharge: list medications with dosage, frequency, and purpose.\n\n"

#         "11. **Discharge Details:**\n"
#         "   - Discharge Disposition (e.g., Home, Rehabilitation, Skilled Nursing Facility)\n"
#         "   - Primary Diagnosis: {question}\n"
#         "   - Secondary Diagnoses: [From context or 'Information not available']\n"
#         "   - Condition at Discharge: [Describe clearly, not as a placeholder]\n\n"

#         "12. **Patient Instructions:**\n"
#         "   Provide actual discharge instructions in natural, patient-friendly language "
#         "about medications, follow-up, diet, and activity.\n\n"

#         "13. **Follow-up Plan:**\n"
#         "   Mention the next steps, including follow-up appointments, tests, or consultations.\n\n"
        
#         "---\n\n"
#         "**Context (MIMIC-IV Notes):**\n{context}\n\n"
#         "Return only the final formatted discharge summary as normal text. Do not include markdown code fences or language tags like ```json or ```."

#         "Now, write one complete discharge summary for this hospitalization related to {question}, using the above structure. "
#         "Do not provide examples or multiple summaries. Write only the final, filled summary."
#     ),
#     input_variables=["context", "question"],
# )

def get_discharge_summary_prompt():
    """
    Returns a refined prompt template that encourages longer,
    better-connected discharge summaries.
    """
    return PromptTemplate(
        template=(
        "You are a clinical summarization model trained on MIMIC-IV notes. "
        "Write a comprehensive, cohesive hospital discharge summary based only on the provided context. "
        "Use formal clinical language and full paragraphs—not lists. "
        "If explicit details are missing, infer logical continuity from the notes "
        "without inventing unrealistic data.\n\n"

        "Your goal is a single, complete discharge summary that reads like a physician-written note. "
        "Include clear transitions between sections so the story flows naturally.\n\n"

        "Structure to follow (expand each part fully):\n\n"

        "1. **Patient Information:** key demographics and service details.\n"
        "2. **Chief Complaint:** concise reason for admission.\n"
        "3. **Major Procedures:** all procedures or 'None'.\n"
        "4. **History of Present Illness:** 3–5 sentences describing onset, course, and precipitating factors for {question}.\n"
        "5. **Past Medical, Social, and Family History:** three short paragraphs (medical / social / family).\n"
        "6. **Physical Examination:** admission vs discharge comparison.\n"
        "7. **Pertinent Results:** labs and imaging, interpreted in words.\n"
        "8. **Hospital Course:** at least 8–10 sentences narrating key events, diagnostics, treatments, and progress.\n"
        "9. **Chronic and Transitional Issues:** summarize ongoing problems and discharge plans.\n"
        "10. **Medications:** admission vs discharge lists with brief purposes.\n"
        "11. **Discharge Details:** disposition, primary & secondary diagnoses, condition at discharge.\n"
        "12. **Patient Instructions:** patient-friendly summary of care and precautions.\n"
        "13. **Follow-up Plan:** who and when to see next.\n\n"

        "---\n\n"
        "**Context (MIMIC-IV Notes):**\n{context}\n\n"
        "Now write one coherent discharge summary focused on {question}. "
        "Do not include markdown fences or extra commentary; output plain text only."
        ),
        input_variables=["context", "question"],
    )



# def get_llm():
#     """Initializes and returns the MedGemma LLM connection."""
#     llm = ChatOpenAI(
#         model_name="medgemma-27b-it",
#         base_url="https://lancinate-persuasive-paxton.ngrok-free.dev/v1",
#         api_key="not-needed",
#         temperature=0.3
#     )
#     return llm 

def get_llm():
    """
    Initializes and returns the MedGemma LLM connection for generating 
    high-fidelity clinical discharge summaries.
    """
    llm = ChatOpenAI(
        model_name="medgemma-27b-it",
        base_url="https://lancinate-persuasive-paxton.ngrok-free.dev/v1",
        api_key="not-needed",
        temperature=0.35,         # lower for factual, deterministic output
        top_p=0.95,                # balanced lexical diversity
        # max_tokens=2048,          # allows full-length summaries
        frequency_penalty=0.3,    # discourages redundant phrasing
        presence_penalty=0.2,     # mild variation encouragement
        request_timeout=180       # prevents timeout for long completions
    )
    return llm
