"""
Streamlit Web Application: AI Research Paper Summarizer & RAG Assistant
Integrates LangChain, FAISS, ChromaDB, Hugging Face, Llama 3 & Mistral.
Features direct drag-and-drop research paper PDF upload with automated end-to-end RAG processing.
"""

import os
import sys
import time
import io
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configure page layout and aesthetics
st.set_page_config(
    page_title="AI Research Paper Summarizer (RAG)",
    layout="wide",
    initial_sidebar_state="expanded"
)

from src.ingestion.pdf_loader import PDFLoader, PaperDocument
from src.ingestion.arxiv_fetcher import ArxivFetcher
from src.ingestion.text_splitter import AcademicTextSplitter
from src.embeddings.embedding_manager import EmbeddingManager, SUPPORTED_EMBEDDING_MODELS
from src.vectorstores.store_manager import VectorStoreManager
from src.llm.llm_provider import LLMProvider, get_llm_provider
from src.rag.qa_chain import ResearchPaperQA
from src.rag.summarizer import PaperSummarizer
from src.rag.benchmark import VectorStoreBenchmark

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1E88E5 0%, #7B1FA2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #64748B;
        font-size: 1.05rem;
        margin-bottom: 1.2rem;
    }
    .upload-box {
        background: linear-gradient(135deg, #F8FAFC 0%, #EFF6FF 100%);
        border: 2px dashed #3B82F6;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
    .active-paper-card {
        background: #F0FDF4;
        border: 1px solid #86EFAC;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .citation-box {
        background-color: #F0F9FF;
        border-left: 4px solid #0284C7;
        padding: 0.8rem 1rem;
        border-radius: 6px;
        margin: 0.5rem 0;
        font-size: 0.95rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0 0;
        padding: 10px 18px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State
if "store_manager" not in st.session_state:
    st.session_state.store_manager = None
if "current_paper" not in st.session_state:
    st.session_state.current_paper = None
if "indexed_chunks" not in st.session_state:
    st.session_state.indexed_chunks = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "summary_cache" not in st.session_state:
    st.session_state.summary_cache = {}
if "last_processed_file" not in st.session_state:
    st.session_state.last_processed_file = None


def process_and_index_document(source, file_name, chunk_size, chunk_overlap, embedding_choice, active_engine):
    """Core pipeline to parse, chunk, embed, and index a research paper into FAISS & ChromaDB."""
    loader = PDFLoader()
    paper_doc = loader.load(source, file_name=file_name)
    st.session_state.current_paper = paper_doc

    # Text Splitting
    splitter = AcademicTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_paper(paper_doc)
    st.session_state.indexed_chunks = chunks

    # Embeddings and Vector Stores
    embeddings = EmbeddingManager.get_embeddings(embedding_choice)
    store_mgr = VectorStoreManager(embeddings=embeddings)
    store_mgr.set_active_engine(active_engine)
    store_mgr.index_documents(chunks, collection_name=paper_doc.file_name, target="both")

    st.session_state.store_manager = store_mgr
    st.session_state.chat_history = []
    st.session_state.summary_cache = {}
    st.session_state.last_processed_file = file_name
    return paper_doc, chunks


# Sidebar Controls: Engine & Advanced Config
with st.sidebar:
    st.markdown("### Engine Settings")

    active_vector_engine = st.selectbox(
        "Active Vector Engine",
        ["FAISS", "ChromaDB"],
        index=0,
        help="FAISS: Ultra-fast in-memory dense vector search. ChromaDB: Persistent vector store with collection metadata."
    )

    embedding_choice = st.selectbox(
        "Hugging Face Embeddings",
        list(SUPPORTED_EMBEDDING_MODELS.keys()),
        index=0,
        help="Local dense embeddings via sentence-transformers"
    )

    with st.expander("Chunking Hyperparameters", expanded=False):
        chunk_size = st.slider("Chunk Size (characters)", 400, 1500, 900, step=50)
        chunk_overlap = st.slider("Chunk Overlap (characters)", 50, 300, 150, step=25)
        top_k = st.slider("Top-K Retrieved Passages", 2, 10, 5)

    st.markdown("---")
    st.markdown("### Open-Source LLM Provider")

    llm_backend = st.selectbox(
        "LLM Backend",
        ["Auto-detect", "Ollama (Local)", "Groq (Cloud)", "Built-in Synthesizer"],
        index=0
    )

    backend_mapped = {
        "Auto-detect": "auto",
        "Ollama (Local)": "Ollama",
        "Groq (Cloud)": "Groq",
        "Built-in Synthesizer": "Built-in"
    }[llm_backend]

    selected_model = None
    if backend_mapped == "Ollama":
        selected_model = st.selectbox("Ollama Model", ["llama3", "llama3.2", "mistral", "mixtral", "phi3"])
    elif backend_mapped == "Groq":
        selected_model = st.selectbox("Groq Model", ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"])
        groq_api_key = st.text_input("Groq API Key (Optional)", type="password", value=os.environ.get("GROQ_API_KEY", ""))
        if groq_api_key:
            os.environ["GROQ_API_KEY"] = groq_api_key

    st.markdown("---")
    st.markdown("### Alternate Sourcing")
    with st.expander("Download from arXiv by ID/URL", expanded=False):
        arxiv_input = st.text_input("arXiv ID (e.g. 1706.03762)", placeholder="1706.03762")
        if st.button("Fetch & Process arXiv Paper", use_container_width=True):
            if arxiv_input:
                fetcher = ArxivFetcher()
                with st.spinner("Downloading paper from arXiv..."):
                    try:
                        downloaded_path = fetcher.download_pdf(arxiv_input)
                        process_and_index_document(
                            downloaded_path,
                            os.path.basename(downloaded_path),
                            chunk_size,
                            chunk_overlap,
                            embedding_choice,
                            active_vector_engine
                        )
                        st.success(f"Loaded arXiv: {os.path.basename(downloaded_path)}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error fetching from arXiv: {e}")

    with st.expander("Load Landmark AI Papers", expanded=False):
        sample_dir = "data/sample_papers"
        available_samples = {
            "Attention Is All You Need (Transformers)": "attention_is_all_you_need.pdf",
            "LoRA: Low-Rank Adaptation of LLMs": "lora_low_rank_adaptation.pdf",
            "Retrieval-Augmented Generation (RAG)": "rag_retrieval_augmented_generation.pdf",
            "Mistral 7B (Sliding Window Attention)": "mistral_7b.pdf",
        }
        selected_sample = st.selectbox("Choose Landmark Paper", list(available_samples.keys()))
        if st.button("Load This Landmark Paper", use_container_width=True):
            path = os.path.join(sample_dir, available_samples[selected_sample])
            if os.path.exists(path):
                with st.spinner("Processing landmark paper..."):
                    process_and_index_document(
                        path,
                        available_samples[selected_sample],
                        chunk_size,
                        chunk_overlap,
                        embedding_choice,
                        active_vector_engine
                    )
                st.success("Loaded landmark paper!")
                st.rerun()


# ==========================================
# MAIN DASHBOARD AREA
# ==========================================

st.markdown('<div class="main-title">AI Research Paper Summarizer (RAG)</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload any research paper (PDF) to chunk, embed in FAISS & ChromaDB, and generate grounded summaries & answers using Llama 3 / Mistral.</div>', unsafe_allow_html=True)

# ------------------------------------------
# PROMINENT CUSTOM PAPER UPLOAD SECTION
# ------------------------------------------
st.markdown("### Upload Research Paper")

col_upload, col_upload_btn = st.columns([3, 1])
with col_upload:
    uploaded_pdf = st.file_uploader(
        "Upload any Research Paper PDF (from your computer)",
        type=["pdf"],
        help="Upload any scientific paper PDF to extract sections, embed vectors, and chat with citations."
    )

with col_upload_btn:
    st.write("") # spacing
    st.write("")
    manual_reprocess = st.button("Process / Re-index", type="primary", use_container_width=True)

# Auto-process uploaded file when a new file is uploaded
if uploaded_pdf is not None:
    file_bytes_name = f"{uploaded_pdf.name}_{uploaded_pdf.size}"
    if st.session_state.last_processed_file != file_bytes_name or manual_reprocess:
        with st.spinner(f"Processing '{uploaded_pdf.name}': extracting text, chunking sections, computing embeddings, and building FAISS & ChromaDB indices..."):
            try:
                # Read bytes and process
                pdf_bytes = io.BytesIO(uploaded_pdf.getvalue())
                paper_doc, chunks = process_and_index_document(
                    source=pdf_bytes,
                    file_name=uploaded_pdf.name,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    embedding_choice=embedding_choice,
                    active_engine=active_vector_engine
                )
                st.session_state.last_processed_file = file_bytes_name
                st.success(f"Successfully ingested & indexed **'{paper_doc.title}'** ({len(chunks)} chunks across {paper_doc.total_pages} pages) into FAISS & ChromaDB.")
            except Exception as e:
                st.error(f"Error processing uploaded PDF: {e}")

# If no paper loaded at all yet, load default landmark paper so app isn't blank
if st.session_state.store_manager is None:
    default_paper_path = "data/sample_papers/attention_is_all_you_need.pdf"
    if os.path.exists(default_paper_path):
        with st.spinner("Initializing system with default landmark paper ('Attention Is All You Need')..."):
            process_and_index_document(
                source=default_paper_path,
                file_name="attention_is_all_you_need.pdf",
                chunk_size=900,
                chunk_overlap=150,
                embedding_choice=embedding_choice,
                active_engine=active_vector_engine
            )


# ------------------------------------------
# ACTIVE PAPER METRIC CARDS
# ------------------------------------------
paper = st.session_state.current_paper
store_mgr = st.session_state.store_manager
if store_mgr:
    store_mgr.set_active_engine(active_vector_engine)

if paper:
    st.markdown(f"""
    <div class="active-paper-card">
        <span style="font-size: 1.15rem; font-weight: 700; color: #1E293B;">Active Paper: {paper.title}</span><br>
        <span style="color: #475569; font-size: 0.92rem;"><b>File:</b> {paper.file_name} &nbsp;|&nbsp; <b>Authors:</b> {paper.authors} &nbsp;|&nbsp; <b>Pages:</b> {paper.total_pages} &nbsp;|&nbsp; <b>Total Chunks:</b> {len(st.session_state.indexed_chunks)} &nbsp;|&nbsp; <b>Active Engine:</b> {active_vector_engine}</span>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Pages", f"{paper.total_pages}")
    with col2:
        st.metric("Vector Chunks", f"{len(st.session_state.indexed_chunks)}")
    with col3:
        st.metric("Vector Store", active_vector_engine)
    with col4:
        st.metric("Embedding Model", embedding_choice)

st.markdown("---")


# ==========================================
# MAIN FUNCTIONAL TABS
# ==========================================

tab_summary, tab_qa, tab_benchmark, tab_chunks = st.tabs([
    "Grounded Summarizer",
    "Interactive Q&A (RAG)",
    "FAISS vs ChromaDB Benchmark",
    "Vector Store & Chunk Explorer"
])


# ------------------------------------------
# TAB 1: Grounded Paper Summarizer
# ------------------------------------------
with tab_summary:
    st.markdown("### Grounded Research Paper Summarization")
    st.write(f"Generates multi-aspect academic summaries for **{paper.title if paper else 'the paper'}**, strictly grounded in retrieved sections.")

    col_opt1, col_opt2 = st.columns([2, 1])
    with col_opt1:
        summary_mode = st.radio(
            "Summary Scope",
            ["Comprehensive Multi-Section Literature Review", "Executive Overview (1-Minute Read)"],
            horizontal=True
        )
    with col_opt2:
        generate_sum_btn = st.button("Generate Summary", type="primary", use_container_width=True)

    mode_key = "multi_section" if "Comprehensive" in summary_mode else "executive"

    if generate_sum_btn and store_mgr:
        llm = get_llm_provider(backend=backend_mapped, model_name=selected_model)
        summarizer = PaperSummarizer(store_manager=store_mgr, llm_provider=llm)

        with st.spinner(f"Synthesizing {summary_mode.lower()} using {active_vector_engine} and {llm.backend}..."):
            res = summarizer.summarize(paper_title=paper.title, mode=mode_key, engine=active_vector_engine)
            st.session_state.summary_cache[mode_key] = res

    # Display cached summary
    if mode_key in st.session_state.summary_cache:
        cached_res = st.session_state.summary_cache[mode_key]
        st.markdown(f"**Generated via {cached_res.engine_used} in {cached_res.latency_ms} ms** using **{cached_res.aspect_chunks_used} target aspect chunks**.")
        st.markdown(cached_res.summary_text)

        st.download_button(
            label="Download Summary as Markdown (.md)",
            data=cached_res.summary_text,
            file_name=f"{os.path.splitext(paper.file_name)[0]}_summary.md",
            mime="text/markdown"
        )
    else:
        st.info("Click **'Generate Summary'** above to generate an executive brief or comprehensive 5-section literature review of your paper.")


# ------------------------------------------
# TAB 2: Interactive Q&A (RAG)
# ------------------------------------------
with tab_qa:
    st.markdown("### Paper-Specific Grounded Question Answering")
    st.write(f"Ask any question about **{paper.title if paper else 'the paper'}**. Answers cite exact pages, sections, and similarity scores.")

    # Quick prompt buttons
    st.markdown("**Suggested Quick Inquiries:**")
    prompt_cols = st.columns(3)
    quick_queries = [
        "What is the core novel mechanism proposed in this paper?",
        "What datasets, benchmarks, and baselines were evaluated?",
        "What are the main limitations and constraints noted by the authors?"
    ]

    selected_quick_prompt = None
    for idx, prompt_text in enumerate(quick_queries):
        with prompt_cols[idx]:
            if st.button(prompt_text, key=f"quick_{idx}", use_container_width=True):
                selected_quick_prompt = prompt_text

    # User query input
    user_query = st.chat_input("Ask a question about this research paper...")
    active_query = selected_quick_prompt or user_query

    # Display Chat History
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["question"])
        with st.chat_message("assistant"):
            st.markdown(chat["answer"])
            if chat.get("citations"):
                with st.expander(f"{len(chat['citations'])} Verified Citations & Retrieved Chunks ({chat.get('engine', 'FAISS')})"):
                    for c in chat["citations"]:
                        st.markdown(f"**Page {c.page} — {c.section}** *(Cosine Similarity: {c.similarity * 100:.1f}%)*")
                        st.caption(f'"{c.preview}"')

    # Process new query
    if active_query and store_mgr:
        with st.chat_message("user"):
            st.write(active_query)

        llm = get_llm_provider(backend=backend_mapped, model_name=selected_model)
        qa = ResearchPaperQA(store_manager=store_mgr, llm_provider=llm)

        with st.chat_message("assistant"):
            with st.spinner(f"Searching {active_vector_engine} & generating grounded answer with {llm.backend}..."):
                res = qa.ask(question=active_query, k=top_k, engine=active_vector_engine)

            st.markdown(res.answer)

            # Metadata info pill
            st.caption(f"Retrieval: {res.retrieval_latency_ms} ms | Generation: {res.generation_latency_ms} ms | Engine: {res.engine_used} | Top Similarity: {res.top_similarity * 100:.1f}%")

            if res.citations:
                with st.expander(f"{len(res.citations)} Verified Citations & Retrieved Chunks"):
                    for c in res.citations:
                        st.markdown(f"**Page {c.page} — {c.section}** *(Similarity: {c.similarity * 100:.1f}%)*")
                        st.caption(f'"{c.preview}"')

            # Save to history
            st.session_state.chat_history.append({
                "question": active_query,
                "answer": res.answer,
                "citations": res.citations,
                "engine": res.engine_used,
                "latency": res.total_latency_ms
            })


# ------------------------------------------
# TAB 3: FAISS vs ChromaDB Benchmark
# ------------------------------------------
with tab_benchmark:
    st.markdown("### FAISS vs ChromaDB Head-to-Head Benchmark")
    st.write(f"Compare retrieval speed, top similarity scores, and chunk overlap on **{paper.title if paper else 'the paper'}** across both engines.")

    if st.button("Run Vector Store Benchmark", type="primary"):
        if store_mgr:
            benchmarker = VectorStoreBenchmark(store_manager=store_mgr)
            with st.spinner("Benchmarking FAISS vs ChromaDB over test queries..."):
                bm_results = benchmarker.run_benchmark(k=top_k)

            ov = bm_results["overall"]
            df_table = bm_results["summary_table"]

            col_b1, col_b2, col_b3, col_b4 = st.columns(4)
            with col_b1:
                st.metric("FAISS Avg Latency", f"{ov['faiss_avg_latency_ms']} ms")
            with col_b2:
                st.metric("ChromaDB Avg Latency", f"{ov['chroma_avg_latency_ms']} ms")
            with col_b3:
                st.metric("Faster Engine", f"{ov['faster_overall']} ({ov['speedup_factor']}x)")
            with col_b4:
                st.metric("Average Overlap", f"{ov['average_overlap_pct']}%")

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df_table["Query"],
                y=df_table["FAISS Latency (ms)"],
                name="FAISS (ms)",
                marker_color="#4F46E5"
            ))
            fig.add_trace(go.Bar(
                x=df_table["Query"],
                y=df_table["Chroma Latency (ms)"],
                name="ChromaDB (ms)",
                marker_color="#F59E0B"
            ))
            fig.update_layout(
                title=f"Query Retrieval Latency: FAISS vs ChromaDB ({paper.file_name if paper else ''})",
                xaxis_title="Benchmark Query",
                yaxis_title="Latency (milliseconds)",
                barmode="group",
                template="plotly_white",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### Detailed Benchmark Matrix")
            st.dataframe(df_table, use_container_width=True)
        else:
            st.warning("Please upload or select a paper first.")
    else:
        st.info("Click **'Run Vector Store Benchmark'** to measure real-time latency and retrieval consistency between FAISS and ChromaDB.")


# ------------------------------------------
# TAB 4: Vector Store & Chunk Explorer
# ------------------------------------------
with tab_chunks:
    st.markdown("### Vector Store & Chunk Inspector")
    st.write(f"Inspect all indexed chunks, scientific sections, and character distributions for **{paper.title if paper else 'the paper'}**.")

    if st.session_state.indexed_chunks:
        chunks = st.session_state.indexed_chunks

        section_counts = {}
        page_counts = {}
        for c in chunks:
            sec = c.metadata.get("section", "General")
            pg = c.metadata.get("page", 1)
            section_counts[sec] = section_counts.get(sec, 0) + 1
            page_counts[pg] = page_counts.get(pg, 0) + 1

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            df_sec = pd.DataFrame(list(section_counts.items()), columns=["Section", "Chunks"])
            fig_sec = px.pie(df_sec, values="Chunks", names="Section", title="Chunk Distribution by Section", hole=0.35)
            fig_sec.update_layout(height=350)
            st.plotly_chart(fig_sec, use_container_width=True)

        with col_c2:
            df_pg = pd.DataFrame(list(page_counts.items()), columns=["Page", "Chunks"]).sort_values("Page")
            fig_pg = px.bar(df_pg, x="Page", y="Chunks", title="Chunk Distribution Across Pages", color_discrete_sequence=["#0284C7"])
            fig_pg.update_layout(height=350)
            st.plotly_chart(fig_pg, use_container_width=True)

        st.markdown("#### Browse Indexed Chunks")
        selected_section_filter = st.selectbox("Filter by Section", ["All"] + list(section_counts.keys()))

        filtered_chunks = [
            c for c in chunks
            if selected_section_filter == "All" or c.metadata.get("section") == selected_section_filter
        ]

        st.write(f"Displaying **{len(filtered_chunks)}** chunks:")
        for idx, c in enumerate(filtered_chunks[:20]):
            with st.expander(f"Chunk #{c.metadata.get('chunk_index', idx)} — Page {c.metadata.get('page')} ({c.metadata.get('section')}) [{c.metadata.get('char_count')} chars]"):
                st.text(c.page_content)
    else:
        st.info("No paper chunks indexed yet.")
