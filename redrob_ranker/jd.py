from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET


@dataclass(frozen=True)
class JobIntent:
    raw_text: str
    target_titles: tuple[str, ...]
    adjacent_titles: tuple[str, ...]
    negative_titles: tuple[str, ...]
    domain_terms: tuple[str, ...]
    vector_terms: tuple[str, ...]
    production_terms: tuple[str, ...]
    evaluation_terms: tuple[str, ...]
    preferred_locations: tuple[str, ...]
    preferred_countries: tuple[str, ...]
    preferred_experience_min: float = 5.0
    preferred_experience_max: float = 9.0


def extract_docx_text(path: str | Path) -> str:
    docx_path = Path(path)
    with zipfile.ZipFile(docx_path) as archive:
        xml = archive.read("word/document.xml")
    root = ET.fromstring(xml)
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs: list[str] = []
    for paragraph in root.findall(".//w:p", ns):
        text = "".join(node.text or "" for node in paragraph.findall(".//w:t", ns)).strip()
        if text:
            paragraphs.append(text)
    return "\n".join(paragraphs)


def build_job_intent(job_path: str | Path) -> JobIntent:
    text = extract_docx_text(job_path)
    return JobIntent(
        raw_text=text,
        target_titles=(
            "senior ai engineer",
            "lead ai engineer",
            "staff machine learning engineer",
            "senior machine learning engineer",
            "machine learning engineer",
            "ml engineer",
            "applied ml engineer",
            "ai engineer",
            "search engineer",
            "recommendation systems engineer",
            "senior nlp engineer",
            "nlp engineer",
            "senior software engineer (ml)",
            "senior data scientist",
        ),
        adjacent_titles=(
            "data scientist",
            "ai research engineer",
            "junior ml engineer",
            "senior software engineer",
            "backend engineer",
            "data engineer",
            "senior data engineer",
            "software engineer",
        ),
        negative_titles=(
            "marketing manager",
            "sales executive",
            "hr manager",
            "accountant",
            "graphic designer",
            "content writer",
            "operations manager",
            "customer support",
            "civil engineer",
            "mechanical engineer",
        ),
        domain_terms=(
            "semantic search",
            "hybrid search",
            "information retrieval",
            "retrieval",
            "ranking",
            "ranker",
            "recommendation system",
            "recommendation systems",
            "recommender",
            "search relevance",
            "learning to rank",
            "vector search",
            "embeddings",
            "embedding",
            "rag",
            "llm",
            "nlp",
            "candidate matching",
        ),
        vector_terms=(
            "pinecone",
            "weaviate",
            "qdrant",
            "milvus",
            "faiss",
            "elasticsearch",
            "opensearch",
            "vector database",
            "hnsw",
            "sentence-transformers",
            "bge",
            "e5",
        ),
        production_terms=(
            "production",
            "deployed",
            "shipped",
            "real users",
            "serving",
            "scale",
            "latency",
            "monitoring",
            "drift",
            "index refresh",
            "pipeline",
            "on-call",
            "a/b test",
            "ab test",
            "online",
            "offline",
            "product",
        ),
        evaluation_terms=(
            "ndcg",
            "mrr",
            "map",
            "offline-online",
            "relevance labeling",
            "evaluation framework",
            "benchmarks",
            "click-through",
            "ab test",
            "a/b test",
        ),
        preferred_locations=(
            "pune",
            "noida",
            "hyderabad",
            "bangalore",
            "bengaluru",
            "mumbai",
            "delhi",
            "gurgaon",
        ),
        preferred_countries=("india",),
    )
