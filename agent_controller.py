def detect_intent(query: str) -> str:
    q = query.lower()

    if any(w in q for w in ["diagram", "flowchart", "draw", "graph"]):
        return "diagram"

    if any(w in q for w in ["image", "picture", "generate image"]):
        return "image"

    if any(w in q for w in ["summarize", "summary", "brief"]):
        return "summary"

    if any(w in q for w in ["explain", "what is", "how", "why"]):
        return "rag"

    return "chat"
