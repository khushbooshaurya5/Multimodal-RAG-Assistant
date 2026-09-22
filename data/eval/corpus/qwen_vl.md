# Qwen-VL Vision-Language Models

Qwen-VL is a family of open vision-language models from Alibaba Cloud. Qwen2-VL and Qwen2.5-VL accept images and video together with text and generate text, which allows them to describe scenes, read text in images (OCR), interpret charts and diagrams, and answer questions about documents.

Qwen2-VL uses a vision transformer encoder with naive dynamic resolution: an image is split into a variable number of patches proportional to its size instead of being resized to a fixed square, so small text in large documents stays legible. Multimodal rotary position embeddings (M-RoPE) encode positions across time, height and width.

The instruct variants are available in 2B, 7B and 72B parameter sizes. The 2B model can run on a consumer GPU or, slowly, on CPU; the larger models are typically served with vLLM behind an OpenAI-compatible API.

For retrieval systems, a vision-language model is used to convert images into searchable text: a structured description covering the image type, visible text, objects and their relationships retrieves much better than a one-line caption.
