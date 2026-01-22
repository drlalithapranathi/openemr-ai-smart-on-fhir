# OpenEMR-AI

AI tooling for OpenEMR: ambient listening, note summarization, and automated coding.

## SMART on FHIR Architecture

This project implements a SMART on FHIR application that integrates with OpenEMR. The frontend authenticates via OAuth2, retrieves patient context from the FHIR API, and sends audio + clinical data to backend services for processing.

Key flow:
1. Clinician launches app from OpenEMR patient dashboard
2. SMART on FHIR OAuth2 authorization with patient context
3. Browser captures ambient audio during clinical encounter
4. Audio transcribed to text via ASR service
5. Transcript + FHIR data (vitals, labs, conditions, meds) sent to summarization
6. RAG-enhanced model generates SOAP note using disease-specific schemas

## Structure

```
smart-ambient-listening/
├── frontend/                    # SMART on FHIR app - launches from OpenEMR, handles OAuth2, captures audio
├── transcription-service/       # Speech-to-text API gateway, calls Modal GPU for ASR inference
└── rag-text-summarization/      # SOAP note generation - retrieves relevant schemas from vector DB, generates clinical summary
```

## Models

| Service | Model | Status |
|---------|-------|--------|
| ASR (Speech-to-Text) | `[TBD]` | Placeholder |
| Summarization | `[TBD]` | Placeholder |
| RAG Embeddings | `[TBD]` | Placeholder |

## Quick Start

See [smart-ambient-listening/README.md](smart-ambient-listening/README.md)

## License

MPL-2.0 license
