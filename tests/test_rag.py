"""
Unit and Integration Tests for AI Research Paper Summarizer & RAG System.
"""

import os
import unittest
from langchain_core.documents import Document

from src.ingestion.pdf_loader import PDFLoader
from src.ingestion.text_splitter import AcademicTextSplitter
from src.embeddings.embedding_manager import EmbeddingManager
from src.vectorstores.faiss_store import FAISSStore
from src.vectorstores.chroma_store import ChromaStore
from src.vectorstores.store_manager import VectorStoreManager
from src.llm.llm_provider import LLMProvider
from src.rag.qa_chain import ResearchPaperQA
from src.rag.summarizer import PaperSummarizer
from src.rag.benchmark import VectorStoreBenchmark


class TestRAGPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sample_pdf = "data/sample_papers/attention_is_all_you_need.pdf"
        cls.embeddings = EmbeddingManager.get_embeddings("all-MiniLM-L6-v2")

    def test_01_pdf_loader(self):
        """Test PDF parsing and metadata extraction."""
        self.assertTrue(os.path.exists(self.sample_pdf), "Sample PDF should exist")
        loader = PDFLoader()
        paper = loader.load(self.sample_pdf)

        self.assertGreater(paper.total_pages, 0)
        self.assertGreater(paper.total_characters, 1000)
        self.assertIn("Attention", paper.title)
        self.assertIsNotNone(paper.abstract)

    def test_02_text_splitter(self):
        """Test academic text chunking and section tagging."""
        loader = PDFLoader()
        paper = loader.load(self.sample_pdf)

        splitter = AcademicTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = splitter.split_paper(paper)

        self.assertGreater(len(chunks), 10)
        first_chunk = chunks[0]
        self.assertIn("page", first_chunk.metadata)
        self.assertIn("section", first_chunk.metadata)
        self.assertIn("source", first_chunk.metadata)

    def test_03_embeddings(self):
        """Test Hugging Face dense embedding vector dimension."""
        text = "Scaled dot-product attention computes softmax(QK^T / sqrt(d_k))V"
        vec = self.embeddings.embed_query(text)
        self.assertEqual(len(vec), 384)

    def test_04_faiss_vector_store(self):
        """Test FAISS indexing and similarity search."""
        docs = [
            Document(page_content="The Transformer uses multi-head self-attention.", metadata={"page": 1, "section": "Intro"}),
            Document(page_content="LoRA freezes pre-trained weights and adapts matrices.", metadata={"page": 2, "section": "Method"}),
        ]
        store = FAISSStore(self.embeddings, index_dir="data/cache/test_faiss")
        res_info = store.build_index(docs, collection_name="test_attention")
        self.assertEqual(res_info["count"], 2)

        results = store.similarity_search_with_score("self-attention transformer", k=1)
        self.assertEqual(len(results), 1)
        self.assertIn("Transformer", results[0][0].page_content)
        self.assertGreater(results[0][1], 0.0)

    def test_05_chroma_vector_store(self):
        """Test ChromaDB persistent collection and search."""
        docs = [
            Document(page_content="The Transformer uses multi-head self-attention.", metadata={"page": 1, "section": "Intro"}),
            Document(page_content="LoRA freezes pre-trained weights and adapts matrices.", metadata={"page": 2, "section": "Method"}),
        ]
        store = ChromaStore(self.embeddings, persist_dir="data/cache/test_chroma")
        res_info = store.build_index(docs, collection_name="test_col")
        self.assertEqual(res_info["count"], 2)

        results = store.similarity_search_with_score("self-attention transformer", k=1)
        self.assertEqual(len(results), 1)
        self.assertIn("Transformer", results[0][0].page_content)
        self.assertGreater(results[0][1], 0.0)

    def test_06_store_manager_benchmark(self):
        """Test VectorStoreManager dual-indexing and side-by-side benchmarking."""
        docs = [
            Document(page_content="Self-attention connects all positions in a sequence.", metadata={"page": 3, "section": "Methodology"}),
            Document(page_content="Experiments on WMT 2014 English-to-German translation achieved 28.4 BLEU.", metadata={"page": 8, "section": "Results"}),
        ]
        mgr = VectorStoreManager(self.embeddings)
        mgr.index_documents(docs, collection_name="test_dual", target="both")

        # Test search with FAISS
        faiss_res = mgr.similarity_search_with_score("BLEU score translation", k=1, engine="FAISS")
        self.assertEqual(len(faiss_res), 1)
        self.assertIn("BLEU", faiss_res[0][0].page_content)

        # Test search with ChromaDB
        chroma_res = mgr.similarity_search_with_score("BLEU score translation", k=1, engine="ChromaDB")
        self.assertEqual(len(chroma_res), 1)
        self.assertIn("BLEU", chroma_res[0][0].page_content)

        # Test benchmark
        bm = mgr.benchmark("BLEU score", k=1)
        self.assertIn("faiss", bm)
        self.assertIn("chroma", bm)
        self.assertIn("overlap_percentage", bm)

    def test_07_grounded_qa(self):
        """Test grounded Q&A with citations and latency recording."""
        docs = [
            Document(page_content="The model achieved 28.4 BLEU on the WMT 2014 English-to-German task.", metadata={"page": 8, "section": "Results", "chunk_index": 12, "source": "paper.pdf"}),
        ]
        mgr = VectorStoreManager(self.embeddings)
        mgr.index_documents(docs, collection_name="test_qa", target="both")

        llm = LLMProvider(backend="Built-in")
        qa = ResearchPaperQA(store_manager=mgr, llm_provider=llm)

        res = qa.ask("What BLEU score was achieved on English-to-German?", k=1)
        self.assertIsNotNone(res.answer)
        self.assertGreater(len(res.citations), 0)
        self.assertEqual(res.citations[0].page, 8)
        self.assertGreater(res.top_similarity, 0.4)

    def test_08_summarizer(self):
        """Test paper summarizer for executive and multi-section modes."""
        docs = [
            Document(page_content="We propose the Transformer, a model architecture eschewing recurrence.", metadata={"page": 1, "section": "Abstract", "chunk_index": 0}),
            Document(page_content="The Transformer achieves 28.4 BLEU on English-to-German translation.", metadata={"page": 7, "section": "Results", "chunk_index": 10}),
        ]
        mgr = VectorStoreManager(self.embeddings)
        mgr.index_documents(docs, collection_name="test_sum", target="both")

        llm = LLMProvider(backend="Built-in")
        summarizer = PaperSummarizer(store_manager=mgr, llm_provider=llm)

        exec_sum = summarizer.summarize(paper_title="Attention Is All You Need", mode="executive")
        self.assertIn("Executive Overview", exec_sum.summary_text)

        multi_sum = summarizer.summarize(paper_title="Attention Is All You Need", mode="multi_section")
        self.assertIn("1. Problem Statement", multi_sum.summary_text)


if __name__ == "__main__":
    unittest.main()
