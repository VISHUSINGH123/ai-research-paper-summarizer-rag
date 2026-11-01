"""
Command-Line Interface (CLI) for AI Research Paper Summarizer & RAG System.
Allows ingesting papers, querying with citations, generating summaries, and running vector benchmarks.
"""

import os
import sys
import argparse
import logging

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

from src.ingestion.pdf_loader import PDFLoader
from src.ingestion.arxiv_fetcher import ArxivFetcher
from src.ingestion.text_splitter import AcademicTextSplitter
from src.embeddings.embedding_manager import get_embedding_model
from src.vectorstores.store_manager import VectorStoreManager
from src.llm.llm_provider import get_llm_provider
from src.rag.qa_chain import ResearchPaperQA
from src.rag.summarizer import PaperSummarizer
from src.rag.benchmark import VectorStoreBenchmark

console = Console(force_terminal=True, legacy_windows=False)
logging.basicConfig(level=logging.WARNING)


def get_pipeline(paper_path: str, embedding_model_name: str = "all-MiniLM-L6-v2"):
    """Load, chunk, and index a paper into FAISS and ChromaDB."""
    with console.status(f"[cyan]Loading PDF '{paper_path}'..."):
        loader = PDFLoader()
        paper = loader.load(paper_path)

    console.print(f"[green]Loaded:[/green] {paper.title} ({paper.total_pages} pages, {paper.total_characters:,} chars)")

    with console.status("[cyan]Chunking text preserving academic sections..."):
        splitter = AcademicTextSplitter(chunk_size=900, chunk_overlap=150)
        chunks = splitter.split_paper(paper)

    console.print(f"[green]Generated:[/green] {len(chunks)} contextual chunks.")

    with console.status("[cyan]Embedding with Hugging Face and indexing into FAISS & ChromaDB..."):
        embeddings = get_embedding_model(embedding_model_name)
        store_mgr = VectorStoreManager(embeddings=embeddings)
        store_mgr.index_documents(chunks, collection_name=paper.file_name, target="both")

    console.print("[bold green]Vector indexing complete for both FAISS and ChromaDB![/bold green]\n")
    return paper, chunks, store_mgr


def cmd_list_samples(args):
    sample_dir = "data/sample_papers"
    if not os.path.exists(sample_dir):
        console.print("[yellow]No sample directory found. Run 'python download_samples.py' first.[/yellow]")
        return

    files = [f for f in os.listdir(sample_dir) if f.endswith(".pdf")]
    table = Table(title="Curated Landmark Research Papers", show_header=True)
    table.add_column("File Name", style="cyan")
    table.add_column("Size", justify="right")
    table.add_column("Path", style="dim")

    for f in files:
        path = os.path.join(sample_dir, f)
        size = f"{os.path.getsize(path) / 1024 / 1024:.2f} MB"
        table.add_row(f, size, path)

    console.print(table)


def cmd_ingest(args):
    paper, chunks, store_mgr = get_pipeline(args.paper)
    console.print(Panel(f"[bold]Title:[/bold] {paper.title}\n[bold]Authors:[/bold] {paper.authors}\n[bold]Pages:[/bold] {paper.total_pages}\n[bold]Chunks:[/bold] {len(chunks)}", title="Ingestion Details"))


def cmd_qa(args):
    paper, chunks, store_mgr = get_pipeline(args.paper)
    llm = get_llm_provider(backend=args.backend, model_name=args.model)
    qa_chain = ResearchPaperQA(store_manager=store_mgr, llm_provider=llm)

    console.print(f"\n[bold yellow]Question:[/bold yellow] {args.query}")
    console.print(f"[dim]Vector Engine: {args.engine} | LLM Backend: {llm.backend} ({llm.model_name})[/dim]\n")

    with console.status("[cyan]Retrieving relevant passages and synthesizing answer..."):
        result = qa_chain.ask(question=args.query, k=args.k, engine=args.engine)

    console.print(Panel(Markdown(result.answer), title=f"Grounded Answer (Top Sim: {result.top_similarity * 100:.1f}%)"))

    # Print citations table
    if result.citations:
        table = Table(title="Source Citations & Snippets", show_header=True)
        table.add_column("Page", style="bold cyan", width=8)
        table.add_column("Section", style="green", width=18)
        table.add_column("Sim Score", justify="right", width=10)
        table.add_column("Snippet Preview", style="dim")

        for c in result.citations:
            table.add_row(f"Page {c.page}", c.section, f"{c.similarity * 100:.1f}%", c.preview)

        console.print(table)
    console.print(f"[dim]Latency: Retrieval {result.retrieval_latency_ms}ms | Generation {result.generation_latency_ms}ms | Total {result.total_latency_ms}ms[/dim]\n")


def cmd_summarize(args):
    paper, chunks, store_mgr = get_pipeline(args.paper)
    llm = get_llm_provider(backend=args.backend, model_name=args.model)
    summarizer = PaperSummarizer(store_manager=store_mgr, llm_provider=llm)

    console.print(f"[dim]Summarizing: {paper.title} (Mode: {args.mode}, Engine: {args.engine})[/dim]\n")

    with console.status("[cyan]Performing targeted multi-aspect RAG summarization..."):
        result = summarizer.summarize(paper_title=paper.title, mode=args.mode, engine=args.engine)

    console.print(Panel(Markdown(result.summary_text), title=f"Summary: {result.paper_title} ({result.mode.upper()})"))
    console.print(f"[dim]Total time: {result.latency_ms}ms | Aspect chunks aggregated: {result.aspect_chunks_used}[/dim]\n")


def cmd_benchmark(args):
    paper, chunks, store_mgr = get_pipeline(args.paper)
    benchmarker = VectorStoreBenchmark(store_manager=store_mgr)

    console.print("[cyan]Running automated FAISS vs ChromaDB performance evaluation...[/cyan]\n")
    report = benchmarker.run_benchmark(k=args.k)

    console.print(report["summary_table"].to_string(index=False))

    ov = report["overall"]
    console.print(Panel(
        f"[bold]FAISS Latency:[/bold] {ov['faiss_avg_latency_ms']} ms\n"
        f"[bold]ChromaDB Latency:[/bold] {ov['chroma_avg_latency_ms']} ms\n"
        f"[bold]Faster Engine:[/bold] [green]{ov['faster_overall']}[/green] ({ov['speedup_factor']}x speedup)\n"
        f"[bold]Average Chunk Overlap:[/bold] {ov['average_overlap_pct']}%\n"
        f"[bold]Total Chunks Evaluated:[/bold] {ov['total_documents_indexed']}",
        title="Benchmark Summary"
    ))


def cmd_arxiv_search(args):
    fetcher = ArxivFetcher()
    with console.status(f"[cyan]Searching arXiv for '{args.query}'..."):
        results = fetcher.search(args.query, max_results=args.limit)

    if not results:
        console.print("[yellow]No papers found for query.[/yellow]")
        return

    table = Table(title=f"arXiv Results for '{args.query}'", show_header=True)
    table.add_column("arXiv ID", style="bold cyan")
    table.add_column("Title", style="white")
    table.add_column("Authors", style="green")
    table.add_column("Published", style="dim")

    for r in results:
        authors_str = ", ".join(r["authors"][:2]) + (" et al." if len(r["authors"]) > 2 else "")
        table.add_row(r["id"], r["title"][:60] + ("..." if len(r["title"]) > 60 else ""), authors_str, r["published"])

    console.print(table)


def cmd_arxiv_download(args):
    fetcher = ArxivFetcher()
    with console.status(f"[cyan]Downloading arXiv paper '{args.id}'..."):
        path = fetcher.download_pdf(args.id)
    console.print(f"[bold green]Saved to:[/bold green] {path}")


def main():
    parser = argparse.ArgumentParser(description="AI Research Paper Summarizer & RAG Assistant")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # list-samples
    subparsers.add_parser("list-samples", help="List available landmark research papers")

    # ingest
    p_ingest = subparsers.add_parser("ingest", help="Ingest, chunk, embed, and index a PDF")
    p_ingest.add_argument("--paper", required=True, help="Path to PDF paper")

    # qa
    p_qa = subparsers.add_parser("qa", help="Ask a question against a paper")
    p_qa.add_argument("--paper", required=True, help="Path to PDF paper")
    p_qa.add_argument("--query", required=True, help="Question to ask")
    p_qa.add_argument("--engine", choices=["FAISS", "ChromaDB"], default="FAISS", help="Vector store engine")
    p_qa.add_argument("--backend", default="auto", choices=["auto", "Ollama", "Groq", "Built-in"], help="LLM backend")
    p_qa.add_argument("--model", default=None, help="LLM model name (e.g. llama3, mistral)")
    p_qa.add_argument("-k", type=int, default=5, help="Top-K chunks to retrieve")

    # summarize
    p_sum = subparsers.add_parser("summarize", help="Generate structured summary of a paper")
    p_sum.add_argument("--paper", required=True, help="Path to PDF paper")
    p_sum.add_argument("--mode", choices=["executive", "multi_section"], default="multi_section", help="Summary mode")
    p_sum.add_argument("--engine", choices=["FAISS", "ChromaDB"], default="FAISS", help="Vector store engine")
    p_sum.add_argument("--backend", default="auto", choices=["auto", "Ollama", "Groq", "Built-in"], help="LLM backend")
    p_sum.add_argument("--model", default=None, help="LLM model name")

    # benchmark
    p_bm = subparsers.add_parser("benchmark", help="Benchmark FAISS vs ChromaDB retrieval")
    p_bm.add_argument("--paper", required=True, help="Path to PDF paper")
    p_bm.add_argument("-k", type=int, default=5, help="Top-K to retrieve")

    # arxiv-search
    p_as = subparsers.add_parser("arxiv-search", help="Search arXiv for papers")
    p_as.add_argument("--query", required=True, help="Search keywords")
    p_as.add_argument("--limit", type=int, default=5, help="Max results")

    # arxiv-download
    p_ad = subparsers.add_parser("arxiv-download", help="Download paper from arXiv by ID or URL")
    p_ad.add_argument("--id", required=True, help="arXiv ID (e.g. 1706.03762)")

    args = parser.parse_args()

    commands = {
        "list-samples": cmd_list_samples,
        "ingest": cmd_ingest,
        "qa": cmd_qa,
        "summarize": cmd_summarize,
        "benchmark": cmd_benchmark,
        "arxiv-search": cmd_arxiv_search,
        "arxiv-download": cmd_arxiv_download,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
