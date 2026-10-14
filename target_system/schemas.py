"""
Pydantic schemas for the Target QA RAG System.
Defines structured input and output representations ensuring deterministic parsing.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Input payload for the QA RAG System."""
    query: str = Field(..., description="The user query or adversarial prompt.")
    context_override: Optional[str] = Field(
        default=None, 
        description="Optional explicit context override (e.g. injected distractor)."
    )


class RagResponse(BaseModel):
    """
    Structured output produced by the Target System.
    Must strictly conform to the evaluation harness requirement:
    - reasoning: Internal deduction and grounding trail
    - extracted_claims: Explicit list of atomic factual statements in the answer
    - typed_claims: Structured factual claims for deterministic verification
    - answer: Final synthesized answer presented to the user
    - supporting_quote: Exact quote from the retrieved corpus supporting the answer
    - source_doc_id: Corpus document id containing the supporting quote
    - confidence_score: Calibrated uncertainty metric (0.0 to 1.0)
    """
    reasoning: str = Field(
        ..., 
        description="Chain-of-thought grounding and deduplication rationale."
    )
    extracted_claims: List[str] = Field(
        ..., 
        description="Atomic factual propositions contained in the answer for fact-checking."
    )
    typed_claims: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Structured factual claims extracted from the answer for deterministic verification."
    )
    answer: str = Field(
        ..., 
        description="The synthesized response to the query."
    )
    supporting_quote: Optional[str] = Field(
        default=None,
        description="Verbatim supporting quote copied from the source document."
    )
    source_doc_id: Optional[str] = Field(
        default=None,
        description="Corpus document id containing the supporting quote."
    )
    confidence_score: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Model confidence score between 0.0 and 1.0."
    )
    rewritten_query: Optional[str] = Field(
        default=None,
        description="The de-noised, intent-isolated query produced by contextual query rewriting."
    )
    retrieved_context: Optional[str] = Field(
        default=None,
        description="The source chunks retrieved from the vector store."
    )
