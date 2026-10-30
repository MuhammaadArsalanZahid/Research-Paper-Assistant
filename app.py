"""app.py - Gradio interface (tabs: Library, Chat, Paper tools). Run:  python app.py"""
import os
import pandas as pd
import gradio as gr

from store import KnowledgeBase
from retriever import retrieve, RERANK_MIN_SCORE, RERANK_STRONG
from llm import LLM, NOT_FOUND_MSG, is_not_found, ensure_citations, sources_block, passages_block
from memory import rewrite_question

KB = KnowledgeBase()      # loads the saved index automatically
MODEL = LLM()             # the language model loads on first use
HEADERS = ["Title", "File", "Pages", "Chunks"]

COMPARE_ASPECTS = [
    ("Objective", "What is the main objective or research question of the paper?"),
    ("Method", "What method or approach does the paper use?"),
    ("Results", "What are the main results and findings of the paper?"),
]


# ------------------------------ helpers ------------------------------
def paper_choices():
    return [(f"{p['title']} ({p['paper_id']})", p["paper_id"]) for p in KB.paper_list()]


def table_frame():
    rows = [[p["title"], p["paper_id"], p["pages"], p["chunks"]] for p in KB.paper_list()]
    return pd.DataFrame(rows, columns=HEADERS)


def dropdown_updates():
    """New choices for: delete, chat-restrict, summarize, compare A, compare B."""
    ch = paper_choices()
    return (
        gr.Dropdown(choices=ch, value=None),
        gr.Dropdown(choices=[("All papers", "")] + ch, value=""),
        gr.Dropdown(choices=ch, value=None),
        gr.Dropdown(choices=ch, value=None),
        gr.Dropdown(choices=ch, value=None),
    )


# ------------------------------ library ------------------------------
def add_papers(files, progress=gr.Progress()):
    if not files:
        return ("Please choose at least one PDF file first.", table_frame(), *dropdown_updates())
    files = files if isinstance(files, list) else [files]
    messages = []
    for n, f in enumerate(files, start=1):
        path = getattr(f, "name", f)
        name = os.path.basename(str(path))
        if not name.lower().endswith(".pdf"):
            messages.append(f"x {name}: not a PDF file, skipped.")
            continue
        progress((n - 1) / len(files), desc=f"Indexing {name} ...")
        try:
            info = KB.add_paper(str(path))
            messages.append(f"ok {name}: '{info['title']}' - {info['pages']} pages, {len(info['chunks'])} chunks.")
        except ValueError as e:
            messages.append(f"x {name}: {e}")
        except Exception as e:
            messages.append(f"x {name}: could not be read ({e}).")
    return ("\n".join(messages), table_frame(), *dropdown_updates())


def delete_paper(paper_id):
    if not paper_id:
        return ("Select a paper to delete first.", table_frame(), *dropdown_updates())
    title = KB.papers.get(paper_id, {}).get("title", paper_id)
    try:
        KB.delete_paper(paper_id)
        msg = f"Deleted '{title}' from the library."
    except Exception as e:
        msg = f"Could not delete: {e}"
    return (msg, table_frame(), *dropdown_updates())


# -------------------------------- chat --------------------------------
def answer_question(question, prior_history, restrict_id, progress=gr.Progress()):
    if not KB.papers:
        return "The library is empty. Please upload papers in the Library tab first."
    try:
        progress(0.25, desc="Understanding your question ...")
        standalone = rewrite_question(MODEL, prior_history, question)   # chat memory
        progress(0.45, desc="Searching the papers ...")
        chunks = retrieve(KB, standalone, paper_id=restrict_id or None)
        # --- debug info in the terminal: always look at the retrieved chunks first ---
        print(f"\n[debug] searched for: {standalone!r}")
        for c in chunks:
            rs = c.get("rerank_score")
            print(f"   re-rank {rs if rs is not None else 'n/a'} | {c['paper_id']} p.{c['page']} ({c['section']})")
        best_score = chunks[0].get("rerank_score") if chunks else None
        if not chunks or best_score is None or best_score < RERANK_MIN_SCORE:
            print("[debug] -> not found (best score below RERANK_MIN_SCORE)")
            return NOT_FOUND_MSG
        progress(0.65, desc="Writing the answer ...")
        answer = MODEL.answer(standalone, chunks)
        print(f"[debug] model said: {answer[:200]!r}")
        if is_not_found(answer):
            if best_score is not None and best_score >= RERANK_STRONG:
                return ("The model could not write a short answer, but these passages look relevant:\n\n"
                        + passages_block(chunks))
            return NOT_FOUND_MSG
        reply = ensure_citations(answer, chunks) + f"\n\n**Sources**\n{sources_block(chunks)}"
        if standalone.strip().lower() != question.strip().lower():
            reply += f"\n\n_Searched for: {standalone}_"
        return reply
    except Exception as e:
        return f"Sorry, something went wrong: {e}"


def chat(message, history, restrict_id):
    history = list(history or [])
    message = (message or "").strip()
    if not message:
        return "", history
    reply = answer_question(message, history, restrict_id)
    history += [{"role": "user", "content": message}, {"role": "assistant", "content": reply}]
    return "", history


# ----------------------------- paper tools -----------------------------
def summarize_paper(paper_id, progress=gr.Progress()):
    if not paper_id:
        return "Select a paper first."
    try:
        progress(0.3, desc="Reading the paper ...")
        title = KB.papers[paper_id]["title"]
        progress(0.6, desc="Writing the summary ...")
        return f"### {title}\n\n" + MODEL.summarize(title, KB.summary_chunks(paper_id))
    except Exception as e:
        return f"Sorry, something went wrong: {e}"


def compare_papers(paper_a, paper_b, progress=gr.Progress()):
    if not paper_a or not paper_b:
        return "Select two papers first."
    if paper_a == paper_b:
        return "Please select two different papers."
    try:
        progress(0.2, desc="Reading Paper A ...")
        ta, tb = KB.papers[paper_a]["title"], KB.papers[paper_b]["title"]
        parts = [f"**Paper A:** {ta}\n\n**Paper B:** {tb}"]
        for n, (aspect, question) in enumerate(COMPARE_ASPECTS, start=1):
            progress(0.2 + 0.25 * n, desc=f"Comparing: {aspect} ...")
            ca = retrieve(KB, question, paper_id=paper_a, top_n=2)
            cb = retrieve(KB, question, paper_id=paper_b, top_n=2)
            parts.append(f"### {aspect}\n" + MODEL.compare_aspect(aspect, ta, ca, tb, cb))
        return "\n\n".join(parts)
    except Exception as e:
        return f"Sorry, something went wrong: {e}"


# -------------------------------- layout --------------------------------
EXAMPLES = [
    "What is the main contribution of these papers?",
    "Which dataset was used for the experiments?",
    "What are the limitations mentioned by the authors?",
]

with gr.Blocks(title="Research Paper Assistant") as demo:
    gr.Markdown("# 📚 Research Paper Assistant\n"
                "Upload research papers, ask questions across all of them, summarize or compare papers. "
                "Every answer shows the paper title and page.")
    with gr.Tabs():
        # ---- Library tab ----
        with gr.Tab("📁 Library"):
            files = gr.File(label="Upload PDF papers (you can select several)",
                            file_count="multiple", file_types=[".pdf"])
            add_btn = gr.Button("Add to library", variant="primary")
            status = gr.Textbox(label="Status", interactive=False, lines=3)
            table = gr.Dataframe(value=table_frame(), label="Indexed papers", interactive=False)
            with gr.Row():
                delete_dd = gr.Dropdown(choices=paper_choices(), label="Select a paper to delete")
                delete_btn = gr.Button("Delete selected paper", variant="stop")

        # ---- Chat tab ----
        with gr.Tab("💬 Chat"):
            restrict = gr.Dropdown(choices=[("All papers", "")] + paper_choices(), value="",
                                   label="Search in")
            try:
                chatbot = gr.Chatbot(label="Conversation", height=430, type="messages")
            except TypeError:                      # newer Gradio versions have no 'type' option
                chatbot = gr.Chatbot(label="Conversation", height=430)
            msg = gr.Textbox(label="Your question", placeholder="Ask a question about your papers...")
            with gr.Row():
                send_btn = gr.Button("Send", variant="primary")
                clear_btn = gr.Button("Clear chat")
            gr.Examples(examples=EXAMPLES, inputs=msg, label="Example questions")

        # ---- Paper tools tab ----
        with gr.Tab("🧰 Paper tools"):
            gr.Markdown("### Summarize a paper")
            sum_dd = gr.Dropdown(choices=paper_choices(), label="Paper")
            sum_btn = gr.Button("Summarize this paper", variant="primary")
            sum_out = gr.Markdown()
            gr.Markdown("### Compare two papers (objective, method, results)")
            with gr.Row():
                cmp_a = gr.Dropdown(choices=paper_choices(), label="Paper A")
                cmp_b = gr.Dropdown(choices=paper_choices(), label="Paper B")
            cmp_btn = gr.Button("Compare papers", variant="primary")
            cmp_out = gr.Markdown()

    lib_outputs = [status, table, delete_dd, restrict, sum_dd, cmp_a, cmp_b]
    add_btn.click(add_papers, inputs=files, outputs=lib_outputs)
    delete_btn.click(delete_paper, inputs=delete_dd, outputs=lib_outputs)
    msg.submit(chat, inputs=[msg, chatbot, restrict], outputs=[msg, chatbot])
    send_btn.click(chat, inputs=[msg, chatbot, restrict], outputs=[msg, chatbot])
    clear_btn.click(lambda: [], outputs=chatbot)
    sum_btn.click(summarize_paper, inputs=sum_dd, outputs=sum_out)
    cmp_btn.click(compare_papers, inputs=[cmp_a, cmp_b], outputs=cmp_out)

if __name__ == "__main__":
    demo.launch()
