"""llm.py - Answer generation, summary and comparison prompts (Qwen instruction model).

Default is the small 0.5B model (works on weak computers). With 16 GB RAM or a GPU you can use
the 1.5B model: set the environment variable RPA_LLM=Qwen/Qwen2.5-1.5B-Instruct
"""
import os
import re
import threading

LLM_NAME = os.environ.get("RPA_LLM", "Qwen/Qwen2.5-0.5B-Instruct")
NOT_FOUND_MSG = "I could not find the answer in the uploaded papers."

ANSWER_SYSTEM = (
    "You are a research assistant. Answer the question using ONLY the context from research "
    "papers below. Give a complete, proper answer in 2-4 sentences. "
    "Cite the paper title and page for every fact, in the format (Paper title, p.5). "
    "If the context does not contain the answer, reply exactly: not found. "
    "Do not give one-word answers; write a full explanation."
)


def format_context(chunks, max_chars=800):
    return "\n\n".join(
        f"[{i}] {c['paper']} (p.{c['page']}, {c['section']})\n{c['text'][:max_chars]}"
        for i, c in enumerate(chunks, start=1))


def is_not_found(text):
    t = (text or "").strip().lower()
    return (not t) or t.startswith("not found") or "not found" in t[:40]


def ensure_citations(answer, chunks):
    """The answer must contain a citation like (Paper title, p.5)."""
    if re.search(r"\([^()]*,\s*p\.\s*\d+", answer) or not chunks:
        return answer
    return f"{answer.rstrip()} ({chunks[0]['paper']}, p.{chunks[0]['page']})"


def sources_block(chunks):
    return "\n".join(
        f"- *{c['paper']}*, p.{c['page']} ({c['section']})" for c in chunks)


def passages_block(chunks, n=2, max_chars=350):
    """Quote the most relevant passages (used when the model cannot write a short answer)."""
    out = []
    for c in chunks[:n]:
        text = c["text"][:max_chars].rstrip() + ("..." if len(c["text"]) > max_chars else "")
        out.append(f"> {text}\n> *{c['paper']}*, p.{c['page']} ({c['section']})")
    return "\n\n".join(out)


class LLM:
    def __init__(self, name=LLM_NAME):
        self.name, self.tok, self.model = name, None, None
        self._lock = threading.Lock()     # Gradio runs requests in threads: load the model only once

    def load(self):
        with self._lock:
            if self.model is not None:
                return
            from transformers import AutoTokenizer, AutoModelForCausalLM
            print(f"[llm] loading {self.name} ...")
            tok = AutoTokenizer.from_pretrained(self.name)
            model = AutoModelForCausalLM.from_pretrained(self.name)
            model.eval()
            # Qwen shares the input embedding weights with the output layer. Some library versions
            # report 'lm_head.weight MISSING' instead of tying them, which gives garbage text.
            # This makes sure they are tied (it does nothing if they already are).
            try:
                emb, head = model.get_input_embeddings(), model.get_output_embeddings()
                if (getattr(model.config, "tie_word_embeddings", False) and head is not None
                        and head.weight.data_ptr() != emb.weight.data_ptr()):
                    head.weight = emb.weight
                    print("[llm] re-tied lm_head to the embedding weights")
            except Exception as e:
                print(f"[llm] could not check weight tying: {e}")
            self.tok, self.model = tok, model

    def chat(self, messages, max_new_tokens=300):
        """Run the model with its chat format. do_sample=False -> deterministic output."""
        self.load()
        import torch
        text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tok(text, return_tensors="pt")
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                      pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def answer(self, question, chunks):
        messages = [
            {"role": "system", "content": ANSWER_SYSTEM},
            {"role": "user", "content": f"Context:\n{format_context(chunks)}\n\nQuestion: {question}"},
        ]
        return self.chat(messages, max_new_tokens=200)

    def summarize(self, title, chunks):
        messages = [
            {"role": "system", "content": (
                "You summarize research papers. Use only the text provided. Write a short "
                "structured summary with these headings: Objective, Method, Results, Conclusion. "
                "One or two sentences each. If something is not in the text, write 'not stated'.")},
            {"role": "user", "content": f"Paper: {title}\n\nText:\n{format_context(chunks)}"},
        ]
        return self.chat(messages, max_new_tokens=150)

    def compare_aspect(self, aspect, title_a, chunks_a, title_b, chunks_b):
        messages = [
            {"role": "system", "content": (
                "You compare two research papers. Use only the text provided. In 2 to 4 sentences "
                "explain how the papers differ (and what they share) for the given aspect. "
                "Mention both paper titles.")},
            {"role": "user", "content": (
                f"Aspect: {aspect}\n\nPaper A: {title_a}\n{format_context(chunks_a)}\n\n"
                f"Paper B: {title_b}\n{format_context(chunks_b)}")},
        ]
        return self.chat(messages, max_new_tokens=120)
