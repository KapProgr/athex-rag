---
title: athex-rag
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: Cited Q&A over Greek ATHEX annual reports
---

# athex-rag: live demo

Ask questions in Greek about the 2024–2025 annual financial reports of ΟΤΕ, Jumbo, ΔΕΗ,
Motor Oil and Εθνική Τράπεζα. Every answer cites the report page it comes from, or says
the information is not in the documents.

This demo runs keyword retrieval (BM25 with Greek stemming, the best configuration in the
evaluation) and a hosted LLM. Code, the 93-question evaluation set and the results:
https://github.com/KapProgr/athex-rag
