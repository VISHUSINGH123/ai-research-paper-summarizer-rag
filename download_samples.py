"""
Download landmark AI research papers for demonstration and offline exploration.
"""

import os
import urllib.request
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

PAPERS = [
    {
        "id": "1706.03762",
        "filename": "attention_is_all_you_need.pdf",
        "title": "Attention Is All You Need",
        "authors": "Vaswani et al. (2017)",
        "topic": "Transformers & Self-Attention Architecture"
    },
    {
        "id": "2005.11401",
        "filename": "rag_retrieval_augmented_generation.pdf",
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "authors": "Lewis et al. (2020)",
        "topic": "Retrieval-Augmented Generation (RAG)"
    },
    {
        "id": "2106.09685",
        "filename": "lora_low_rank_adaptation.pdf",
        "title": "LoRA: Low-Rank Adaptation of Large Language Models",
        "authors": "Hu et al. (2021)",
        "topic": "Parameter-Efficient Fine-Tuning (PEFT)"
    },
    {
        "id": "2310.06825",
        "filename": "mistral_7b.pdf",
        "title": "Mistral 7B",
        "authors": "Jiang et al. (2023)",
        "topic": "Sliding Window Attention & Open-Source LLMs"
    }
]

def download_sample_papers(target_dir: str = "data/sample_papers"):
    os.makedirs(target_dir, exist_ok=True)
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    for paper in PAPERS:
        file_path = os.path.join(target_dir, paper["filename"])
        if os.path.exists(file_path) and os.path.getsize(file_path) > 10000:
            logging.info(f"Already exists: {paper['filename']} ({os.path.getsize(file_path):,} bytes)")
            continue

        url = f"https://arxiv.org/pdf/{paper['id']}.pdf"
        logging.info(f"Downloading '{paper['title']}' from {url}...")
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
                with open(file_path, "wb") as f:
                    f.write(data)
            logging.info(f"Successfully saved {paper['filename']} ({len(data):,} bytes)")
        except Exception as e:
            logging.error(f"Failed to download {paper['filename']}: {e}")

if __name__ == "__main__":
    download_sample_papers()
