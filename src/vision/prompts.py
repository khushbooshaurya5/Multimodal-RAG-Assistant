"""Prompts used for image understanding."""

ANALYSIS_SYSTEM_PROMPT = (
    "You are a meticulous visual analyst. You describe images precisely and never invent "
    "details that are not visible."
)

ANALYSIS_PROMPT = """Analyse this image for a search index. Respond with a single JSON object with exactly these keys:
- "image_kind": one of "document", "diagram", "chart", "table", "screenshot", "photo", "other"
- "description": a dense, factual description (2-6 sentences) covering the subject, layout, visual relationships (what connects to what, what is above/below), and for charts the axes, series and trend
- "visible_text": ALL text legible in the image, transcribed verbatim in reading order (empty string if none)
- "objects": a list of the main objects, components or entities shown (short noun phrases)

Do not add commentary outside the JSON."""

VQA_SYSTEM_PROMPT = (
    "You are a careful assistant answering questions about an image. Base your answer only on "
    "what is visible; if something cannot be determined from the image, say so."
)
