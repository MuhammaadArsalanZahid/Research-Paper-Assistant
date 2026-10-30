# 📚 Research Paper Assistant (RAG)

> 🎥 **Demo video:** https://drive.google.com/file/d/1TxvCxH_-Ih21U6FpHKKsv48gu4s9mQAn/view?usp=sharing

Upload several research papers (PDF), ask questions across all of them, get a summary of one paper, or compare two papers. Every answer shows the **paper title and page** it came from. The knowledge base is saved on disk, so it does not need to be rebuilt each time.

## The problem
Students and researchers read many papers. Searching them one by one is slow, and a language model alone can invent answers. This app uses **Retrieval-Augmented Generation (RAG)**: it finds the most relevant passages in your papers first and lets a small open-source model answer using only those passages, with citations you can check.

## Key features
- Upload multiple PDFs at once; list of indexed papers; delete a paper from the library
- Paper title detection (largest text on page 1) and **section-aware chunking** (Abstract, Introduction, Method, Results, Conclusion...); reference lists are skipped
- Embeddings with `BAAI/bge-small-en-v1.5` (with the BGE query instruction prefix)
- FAISS index and metadata **saved to disk and loaded automatically** at start-up
- **Two-stage retrieval:** 20 candidates from FAISS, re-ranked with a cross-encoder, best 4 kept
- Answers with an instruction model (Qwen2.5-0.5B-Instruct by default) and citations like *(Paper title, p.5)*
- **Chat memory:** the last 3 turns are used to rewrite a follow-up question ("and what dataset did they use?") into a standalone question before searching
- **Paper tools:** "Summarize this paper" and "Compare two papers" (objective, method, results)
- Option to restrict a question to one selected paper
- Loading indicators and friendly error messages throughout the UI
- Gradio interface with separate tabs: Library, Chat, Paper tools

## How it works
```
Upload PDFs -> text per page -> title + section detection -> chunks (800 chars, 150 overlap)
   -> BGE embeddings -> FAISS index + metadata saved on disk

Question -> (rewrite with last 3 turns) -> BGE query embedding -> FAISS top 20
   -> cross-encoder re-ranking -> best 4 chunks -> Qwen answer with citations
```

## Tools and models
| Part | Tool |
|---|---|
| Interface | [Gradio](https://www.gradio.app/) |
| PDF reading | [pypdf](https://pypdf.readthedocs.io/) |
| Embeddings | [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) |
| Vector search | [FAISS](https://github.com/facebookresearch/faiss) (`faiss-cpu`) |
| Re-ranker | [cross-encoder/ms-marco-MiniLM-L-6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) |
| Language model | [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) (default) or [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) |

**Papers used for testing:** The four papers in `sample_papers/` are synthetic test papers written for this project (no license restrictions):
- `aquasense_lora.pdf` — IoT soil moisture monitoring with LoRa (4 pages)
- `leafnet_pruning.pdf` — Lightweight pruning of convolutional networks (4 pages)
- `sleepwell_survey.pdf` — Sleep habits and academic performance survey (4 pages)
- `LLM ACT Different.pdf` — Evaluation awareness of LLMs in Indic languages (19 pages)

Replace these with your own open-access papers (e.g., from arXiv) and list them here with title, link and license.

## Project structure
| File / folder | Purpose |
|---|---|
| `app.py` | Gradio interface |
| `ingest.py` | PDF reading, title and section detection, chunking |
| `store.py` | Embeddings, FAISS saving, loading and deleting |
| `retriever.py` | Search and re-ranking |
| `memory.py` | Chat history and question rewriting |
| `llm.py` | Answer, summary and comparison prompts |
| `evaluation/` | Question lists and the evaluation script |
| `sample_papers/`, `demo/` | Test papers, demo video and screenshots |
| `data/` | Saved index and metadata (created automatically, not uploaded to GitHub) |

## Installation
Tested with **Python 3.11** (very new Python versions can fail to load PyTorch).

```bash
python -m venv venv
venv\Scripts\activate            # Windows   (Linux/Mac: source venv/bin/activate)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python app.py
```
Open the link shown in the terminal (usually `http://127.0.0.1:7860`). The first run downloads the models (about 1.5 GB for the default setup), so keep enough free disk space.

**Weak computer:** the default model is the 0.5B Qwen. **16 GB RAM or a GPU:** use the 1.5B model with `set RPA_LLM=Qwen/Qwen2.5-1.5B-Instruct` (Windows CMD) before `python app.py`.

## How to use
1. **Library tab:** upload PDFs and click *Add to library*. The papers appear in the table; use the dropdown to delete one.
2. **Chat tab:** ask questions. Choose a single paper in *Search in* to restrict the search.
3. **Paper tools tab:** summarize a paper or compare two papers.

## Screenshots
<img width="1914" height="882" alt="Screenshot 2026-10-31 013504" src="https://github.com/user-attachments/assets/5dbeb4aa-e46a-48c5-9d6b-b75802dbd5a3" />
<img width="1917" height="957" alt="Screenshot 2026-10-31 013155" src="https://github.com/user-attachments/assets/448438e9-28b7-4c72-ae05-3b563557a8d1" />
<img width="1880" height="470" alt="Screenshot 2026-10-31 013230" src="https://github.com/user-attachments/assets/44563d03-23de-46a8-a88d-d1037cfd07bf" />

<img width="1914" height="869" alt="Screenshot 2026-10-31 013254" src="https://github.com/user-attachments/assets/944f8ffd-5175-4c79-8fb7-36f91090a63c" />

## Results
Run the evaluation after adding your papers to the library:
```bash
python evaluation/evaluate.py
```
It reports how often the correct paper (and page) is in the final top 4, **with and without re-ranking**.

20 evaluation questions are in `evaluation/questions_sample.json` (15 with known answers, 5 "not in any paper" negatives).

| Setting | Paper hit-rate (top 4) | Page hit-rate (top 4) |
|---|---|---|
| FAISS only (no re-ranking) | paper hit-rate 3/15 (20%) | page hit-rate 2/15 (13%) |
| FAISS + re-ranking | paper hit-rate 3/15 (20%) | page hit-rate 3/15 (20%) |

Follow-up question tests (at least 5): TODO (question, rewritten standalone question, correct?).

Value of `RERANK_MIN_SCORE` used for the "not found" case: TODO.

## Limitations
- Title and section detection use simple rules; unusual paper layouts may get a wrong title or a missing section label.
- PDFs with two columns can mix the reading order of the extracted text.
- Scanned PDFs (images only) are not supported; the app shows a message.
- A small language model can make mistakes, ignore the "not found" rule, or write weak summaries and comparisons. Always check the cited pages.
- TODO: add the failures you really found during testing.

## Future improvements
- Hybrid search (BM25 + FAISS), OCR for scanned papers, a larger model, deployment on Hugging Face Spaces.

## Author
**Muhammad Arsalan Zahid**: [LinkedIn](https://www.linkedin.com/in/muhammad-arsalan-zahid-aa255b330/)

*Built during the AI/ML Internship at Taviq AI Solutions.*
