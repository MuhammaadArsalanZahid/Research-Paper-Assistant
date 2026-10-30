"""memory.py - Chat memory: rewrite a follow-up question into a standalone question
using the last 3 question-answer pairs."""

MAX_TURNS = 3
_SOURCES_MARK = "\n\n**Sources**"


def _text_of(content):
    """Chat messages can hold a plain string or a list of content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return str(content.get("text") or content.get("content") or "")
    if isinstance(content, (list, tuple)):
        return " ".join(_text_of(c) for c in content)
    return ""


def history_to_pairs(history):
    """[{"role","content"}, ...] -> [(question, answer), ...] (last MAX_TURNS only)."""
    pairs, pending = [], None
    for m in history or []:
        role, text = m.get("role"), _text_of(m.get("content"))
        if role == "user":
            pending = text
        elif role == "assistant" and pending is not None:
            pairs.append((pending, text.split(_SOURCES_MARK)[0].strip()))
            pending = None
    return pairs[-MAX_TURNS:]


def rewrite_question(llm, history, question):
    """Return a standalone version of the question (or the original if no history)."""
    pairs = history_to_pairs(history)
    if not pairs:
        return question
    convo = "\n".join(f"User: {q}\nAssistant: {a[:300]}" for q, a in pairs)
    messages = [
        {"role": "system", "content": (
            "You rewrite follow-up questions. Using the conversation, rewrite the last question "
            "so that it is fully standalone: replace words like 'they', 'it', 'that paper' with "
            "what they refer to. Do not answer the question. Output only the rewritten question.")},
        {"role": "user", "content": f"Conversation:\n{convo}\n\nLast question: {question}\n\nStandalone question:"},
    ]
    try:
        out = llm.chat(messages, max_new_tokens=60).strip()
    except Exception:
        return question
    out = out.splitlines()[0].strip().strip('"') if out else ""
    return out if 5 <= len(out) <= 300 else question
