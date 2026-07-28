# RAG Chatbot — Attention Is All You Need

## Objective

Build a minimal Retrieval-Augmented Generation (RAG) pipeline: retrieve relevant passages from a source document and use them as grounded context for an LLM answer. This capstone demonstrates the full loop end-to-end in a single Jupyter notebook.

## Document Used

**"Attention Is All You Need"** — Vaswani et al., 2017 ([arXiv:1706.03762](https://arxiv.org/pdf/1706.03762)). Public, well-scoped, and rich in technical concepts (self-attention, multi-head attention, positional encoding, BLEU results) that make retrieval quality easy to inspect.

The PDF is downloaded fresh at notebook runtime and is **not** committed to the repo.

## Libraries Used

- `pypdf` — PDF text extraction
- `sentence-transformers` (`all-MiniLM-L6-v2`) — 384-dim sentence embeddings, CPU-friendly, no API key
- `faiss-cpu` — local vector index with inner-product / cosine search
- `google-generativeai` — Gemini API client for the generation step (free tier)
- `numpy` — array plumbing

## Methodology

1. **Load** — Download PDF, extract text page-by-page with `pypdf`.
2. **Chunk** — Split into overlapping word windows (~180 words, 30-word overlap) → 41 chunks.
3. **Embed** — `all-MiniLM-L6-v2` → normalized 384-dim vectors.
4. **Index** — FAISS `IndexFlatIP` (inner product on unit vectors ≡ cosine similarity).
5. **Retrieve** — `retrieve(query, k=4)` returns top-k chunks with similarity scores.
6. **Generate** — Build a prompt that clearly separates *context* from *question*, instruct Gemini Flash Lite to answer **only** from the provided context (else say it doesn't know), call the API, return the grounded answer.

## How to Run

```bash
pip install sentence-transformers faiss-cpu pypdf google-generativeai jupyter
export GOOGLE_API_KEY="..."   # required for Task 4 & 5 only
jupyter notebook rag_chatbot.ipynb
```

Get a free Gemini API key at https://aistudio.google.com/apikey. Then Run All. Tasks 1–3 (download → chunk → embed → retrieve) run with no API key. Tasks 4–5 (Gemini generation) require `GOOGLE_API_KEY` in the environment before launching the kernel.

## Example Q&A Output

```
Q: What is the Transformer architecture?
  [1] 0.390 | on self-attention to compute representations of its input and output without using sequence-aligned RNNs or convolution...
  [2] 0.380 | this work we employ h = 8 parallel attention layers, or heads. For each of these we use dk = dv = dmodel/h = 64...
  [3] 0.348 | Table 2: The Transformer achieves better BLEU scores than previous state-of-the-art models on the English-to-German...
```

For a question the document cannot answer (e.g. "What is the capital of France?"), the grounded prompt causes the model to explicitly decline rather than hallucinate — the key RAG property this notebook demonstrates.

## RAG Architecture Note

RAG = **Retrieval** (find the most relevant passages from a trusted source using vector similarity) + **Generation** (hand those passages to an LLM as context and instruct it to answer only from them). This grounds answers in real, cite-able content rather than the model's parametric memory, which is the main reason RAG reduces hallucination for domain- or document-specific questions.

## Conclusion

The notebook implements a complete but minimal RAG pipeline: PDF → chunks → dense embeddings → FAISS retrieval → Gemini generation with a context-only system prompt. It correctly answers technical questions about the Transformer and declines out-of-scope questions. Obvious next steps: hybrid (BM25 + dense) retrieval, cross-encoder re-ranking, semantic chunking that respects section boundaries, and conversation memory for follow-up turns.
