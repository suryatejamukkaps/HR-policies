# HR Policy RAG Assistant

This project is a local RAG-based chatbot that answers questions from a company HR policy document.

The app uses:

- Ollama `llama3.2` as the local LLM
- Ollama `mxbai-embed-large:latest` as the embedding model
- ChromaDB as the local vector database
- LangChain for the RAG pipeline
- Streamlit for the chat interface

The assistant answers only from the HR policy document and avoids using outside knowledge.

---

## Project Purpose

The purpose of this project is to create a local HR policy assistant that can help users quickly ask questions about company policies, such as leave, notice period, holidays, work hours, payroll, employee conduct, and related HR topics.

Instead of manually searching through a long PDF, users can ask questions in a chat interface and get brief answers based on the policy document.

---

## Information Available in the HR Policy Document

The HR policy document includes information about:

- Company introduction and management philosophy
- Business ethics and employee conduct
- Employee relations and grievance redressal
- Disciplinary rules and workplace conduct
- Recruitment, selection, and onboarding
- Buddy policy
- Performance management
- Bonus policy
- Professional development policy
- Separation and notice period
- Employment termination
- Employee categories and records
- Employee benefits
- Holidays and optional holidays
- Sick leave benefits
- Annual/casual leave benefits
- Maternity leave benefits
- Leave without pay
- Employee referral program
- Time keeping and payroll
- Work schedule and attendance
- Dress code
- Personal communication and email usage
- Overtime policy
- Employee conduct and discipline
- Return of company property
- Problem resolution
- Travel and accommodation policy
- Equal employment opportunity policy
- Sexual harassment policy
- Health, safety, smoking, alcohol, and drug policy

The original HR policy PDF was scanned/image-based, so it was converted into a readable text-based PDF before creating embeddings. The source document contains the HR manual content across 50 pages. :contentReference[oaicite:0]{index=0}

---

## Project Structure

```text
HR policies/
│
├── policies/
│   └── productsquads_hr_policies_clean_readable.pdf
│
├── chroma_db/
│
├── app.py
├── vector.py
├── build_db.py
├── requirements.txt
└── README.md
