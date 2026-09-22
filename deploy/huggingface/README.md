---
title: Multimodal RAG Assistant
emoji: 🧠
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 8501
pinned: true
license: mit
short_description: Qwen-VL + Whisper + FAISS RAG over text, images and voice
tags:
  - rag
  - multimodal
  - whisper
  - qwen-vl
  - faiss
---

# Multimodal RAG Assistant

Ask questions by **text, image or voice**. Speech is transcribed with Whisper, images
are read by Qwen2-VL, relevant passages are retrieved from a FAISS index and the answer
cites its evidence with an explicit "insufficient evidence" behaviour.

Try it: upload a PDF or a few notes in the sidebar, click *Index uploaded files*, then
ask a question. Attach a diagram or record a spoken question to exercise the vision and
speech paths. Every answer shows the retrieved sources and an evidence indicator.

This Space runs on CPU. The first request after a cold start downloads the models and
can take a few minutes; Qwen2-VL-2B answers take tens of seconds on the free CPU tier.
For a faster demo switch the Space to a GPU or point `MRAG_VISION_BACKEND` /
`MRAG_LLM_BACKEND` at an OpenAI-compatible endpoint.

Source, tests and evaluation: https://github.com/khushbooshaurya5/Multimodal-RAG-Assistant
