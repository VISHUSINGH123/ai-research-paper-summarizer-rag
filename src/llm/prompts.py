"""
Scientific Prompt Templates for Grounded Paper Summaries and Question Answering.
"""

GROUNDED_QA_PROMPT = """You are an expert AI research scientist and literature review assistant.
Your task is to answer the user's question about the research paper using ONLY the provided context snippets.

STRICT GROUNDING GUIDELINES:
1. Base your answer ENTIRELY on the provided text in the <context> section.
2. If the context does not contain sufficient details to answer the question, state:
   "Based on the retrieved sections of this paper, there is insufficient evidence to answer this question." Do NOT hallucinate.
3. Every factual claim, architecture detail, equation, benchmark, or parameter must cite its source in format: `[Page X - Section Name]`.
4. Be precise, technical, and academically rigorous.

<context>
{context}
</context>

Question: {question}

Provide a well-structured, clear, grounded answer with page citations:"""


EXECUTIVE_SUMMARY_PROMPT = """You are an expert AI research scientist.
Generate an executive, high-level summary of the research paper based ONLY on the provided context excerpts from the paper.

<context>
{context}
</context>

Format your response in Markdown:
### 🎯 Executive Overview
(2-3 paragraphs explaining the core research problem, why previous approaches failed or were limited, and the authors' key insight.)

### 🌟 Key Innovation
(What is the exact novel mechanism, formula, or architecture proposed in this paper?)

### 📊 Principal Findings
(Primary benchmark results, metrics, efficiency gains, or empirical breakthroughs with numbers where available.)

### 💡 Practical Impact
(Why this paper matters for researchers and ML practitioners.)"""


MULTI_SECTION_SUMMARY_PROMPT = """You are an expert AI researcher conducting a thorough literature review.
Produce a comprehensive, rigorous multi-section breakdown of this paper using ONLY the retrieved sections below.

<context>
{context}
</context>

Organize your review strictly into these 5 sections:

### 1. Problem Statement & Motivation
- What fundamental challenge in machine learning / NLP / computer vision does this paper address?
- What are the limitations or bottlenecks of prior work?

### 2. Core Contributions & Novelty
- Enumerate the primary novel contributions (bulleted).
- What makes this formulation or approach distinct from previous baselines?

### 3. Architecture & Methodology
- Detail the algorithmic formulation, mathematical structure, or architecture (e.g., equations, attention formulas, loss functions).
- Explain key hyperparameters or structural components (layers, heads, dimensions, ranks) cited in the text.

### 4. Experimental Setup & Quantitative Results
- What benchmarks, datasets, and baselines were used for evaluation?
- Detail the empirical results, improvements (e.g. BLEU scores, perplexity, speedup, parameter reduction).

### 5. Limitations & Future Directions
- What constraints, computational costs, assumptions, or failure modes did the authors note?
- What directions for future work or extensions were highlighted?

Ground every point with references to pages or sections from the context."""


COMPARATIVE_PROMPT = """You are an AI research analyst comparing two seminal research papers.
Using the provided contexts from Paper A and Paper B, construct a structured comparative analysis.

Context Paper A:
{context_a}

Context Paper B:
{context_b}

Compare the two papers along:
1. Core Problem Addressed
2. Algorithmic / Architectural Difference
3. Computational Efficiency & Resource Requirements
4. Empirical Performance & Breakthroughs
5. Trade-offs and Complementary Use Cases
"""
