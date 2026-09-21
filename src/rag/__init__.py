"""Retrieval-augmented generation: ingestion, context assembly, generation, orchestration."""

from src.rag.context import ContextAssembler, build_citations
from src.rag.evidence import compute_evidence
from src.rag.generator import AnswerGenerator, ExtractiveGenerator, build_generator
from src.rag.ingestion import IngestionPipeline, IngestResult
from src.rag.pipeline import RAGPipeline
from src.rag.service import AssistantService, BackendStatus

__all__ = [
    "AnswerGenerator",
    "AssistantService",
    "BackendStatus",
    "ContextAssembler",
    "ExtractiveGenerator",
    "IngestResult",
    "IngestionPipeline",
    "RAGPipeline",
    "build_citations",
    "build_generator",
    "compute_evidence",
]
