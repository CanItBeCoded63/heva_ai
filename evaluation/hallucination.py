"""
Module A: Principled Hallucination Detection Engine.

================================================================================
CRITICAL DESIGN SPECIFICATION:
================================================================================
Basic string matching fails on paraphrased factual claims, syntactic inversions,
and subtle numerical/temporal hallucinations.

This module implements a principled Natural Language Inference (NLI) entailment
judge using an LLM-as-a-judge protocol. For each atomic claim in output_claims:
1. Presents the extracted claim and the retrieved source_context to the judge.
2. Performs strict entailment checking:
   - ENTAILED: The claim is directly supported and asserted by the source context.
   - CONTRADICTED: The claim directly contradicts source material.
   - UNGROUNDED / HALLUCINATED: The claim introduces new entities, dates, or numbers
     unsupported by the source context.
3. Computes:
   - is_hallucination: bool (True if ANY claim is ungrounded/contradicted)
   - hallucination_score: float (proportion of hallucinated claims)
   - claim_verdicts: detailed per-claim diagnostic breakdown
================================================================================
"""

import re
import json
from difflib import SequenceMatcher
from typing import List, Dict, Any, Tuple, Optional
from target_system.llm_client import LLMClient

_NUMBER_RE = re.compile(r"\d[\d,]*\.?\d*%?")
_MONTH_RE = re.compile(
    r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
    r"aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b",
    re.IGNORECASE,
)


def extract_numbers(text: str) -> set:
    """Extracts numeric tokens, normalized for comparison."""
    if not text:
        return set()
    nums = set()
    for n in _NUMBER_RE.findall(text):
        cleaned = n.replace(",", "").strip(".,%")
        if cleaned:
            nums.add(cleaned)
    return nums


def extract_months(text: str) -> set:
    """Extracts 3-letter lowercase month identifiers."""
    if not text:
        return set()
    return {m.lower()[:3] for m in _MONTH_RE.findall(text)}


def check_span_exists(claim: str, source_text: str, fuzzy_threshold: float = 0.85, window_stride: int = 15) -> bool:
    """
    Layer 1: Checks whether the claim appears verbatim or with high fuzzy similarity in source context.
    Strictly verifies that any numeric tokens in the claim are present in the source text.
    """
    if not claim or not source_text:
        return False

    norm_claim = " ".join(claim.replace("**", "").split())
    norm_source = " ".join(source_text.replace("**", "").split())

    if norm_claim.lower() in norm_source.lower():
        return True

    # Exact numeric containment check
    claim_nums = extract_numbers(norm_claim)
    source_nums = extract_numbers(norm_source)
    if claim_nums and not claim_nums.issubset(source_nums):
        return False

    # Sliding window fuzzy match
    n = len(norm_claim)
    if n == 0:
        return False
    best_ratio = 0.0
    for i in range(0, max(1, len(norm_source) - n), window_stride):
        window = norm_source[i:i + n]
        ratio = SequenceMatcher(None, norm_claim.lower(), window.lower()).ratio()
        best_ratio = max(best_ratio, ratio)
        if best_ratio >= fuzzy_threshold:
            return True

    return best_ratio >= fuzzy_threshold


def check_numeric_date_entailment(claim: str, source_text: str) -> Tuple[bool, Optional[str]]:
    """
    Layer 2: Symbolic numeric and temporal entity containment check.
    Ensures numbers and months in claim are grounded in source text.
    """
    if not claim:
        return False, "Empty claim or source text."

    refusal_markers = [
        "not contain enough information", "does not contain enough information",
        "retrieved context does not contain", "not available in the indexed corpus",
        "insufficient information", "cannot answer", "not in the corpus"
    ]
    if not source_text and any(marker in claim.lower() for marker in refusal_markers):
        return True, None

    if not source_text:
        return False, "Empty claim or source text."

    claim_nums = extract_numbers(claim)
    source_nums = extract_numbers(source_text)
    missing_nums = claim_nums - source_nums
    if missing_nums:
        return False, f"Claim asserts numeric values not present in source context: {missing_nums}"

    claim_months = extract_months(claim)
    source_months = extract_months(source_text)
    missing_months = claim_months - source_months
    if missing_months:
        return False, f"Claim references months not present in source context: {missing_months}"

    return True, None


class HallucinationDetector:
    """
    Evaluates factual claims against source context via claim-level NLI verification.
    """

    def __init__(self, llm_client: LLMClient = None, seed: int = 42):
        self.llm_client = llm_client or LLMClient(seed=seed, temperature=0.0)

    def verify_single_claim(self, claim: str, source_context: str) -> Dict[str, Any]:
        """
        Submits an individual atomic claim to the NLI entailment judge.
        """
        # Fast Layer 2 Check: Symbolic numeric/date containment
        num_ok, num_reason = check_numeric_date_entailment(claim, source_context)
        if not num_ok:
            return {
                "claim": claim,
                "entailed": False,
                "is_hallucination": True,
                "reasoning": f"Layer 2 Numeric Failure: {num_reason}"
            }

        system_prompt = (
            "You are a strict factual entailment judge. You evaluate whether an atomic claim "
            "is entailed by the provided source context.\n"
            "Classification Rules:\n"
            "1. ENTAILED: The claim can be directly derived or inferred strictly from the source.\n"
            "2. HALLUCINATED: The claim asserts facts, dates, entities, or relations NOT supported "
            "or contradicted by the source context.\n\n"
            "Respond strictly in JSON format:\n"
            "{\n"
            "  \"entailed\": true/false,\n"
            "  \"is_hallucination\": true/false,\n"
            "  \"reasoning\": \"precise explanation of why the claim is supported or hallucinated\"\n"
            "}"
        )

        user_prompt = (
            f"Source Context:\n{source_context}\n\n"
            f"Claim:\n{claim}\n\n"
            "Provide your strict entailment verdict in JSON format."
        )

        response_str = self.llm_client.chat_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            json_mode=True
        )

        try:
            verdict = json.loads(response_str)
            entailed = bool(verdict.get("entailed", False))
            if "is_hallucination" in verdict:
                is_hall = bool(verdict["is_hallucination"])
            else:
                is_hall = not entailed
            return {
                "claim": claim,
                "entailed": entailed,
                "is_hallucination": is_hall,
                "reasoning": verdict.get("reasoning", "Evaluated by entailment judge.")
            }
        except Exception:
            return {
                "claim": claim,
                "entailed": False,
                "is_hallucination": True,
                "reasoning": "Judge output could not be parsed as valid JSON."
            }

    def detect_hallucinations(
        self,
        output_claims: List[str],
        source_context: str
    ) -> Tuple[bool, str, List[Dict[str, Any]]]:
        """
        Module A primary interface:
        Evaluates a list of extracted claims against the ground-truth source context.
        
        Returns:
            (is_hallucination, summary_reasoning, per_claim_verdicts)
        """
        if not output_claims:
            return False, "No factual claims asserted.", []

        claim_verdicts = []
        hallucination_count = 0

        for claim in output_claims:
            if not claim.strip():
                continue
            if not source_context and any(marker in claim.lower() for marker in [
                "not contain enough information", "does not contain enough information",
                "retrieved context does not contain", "not available in the indexed corpus",
                "insufficient information", "cannot answer", "not in the corpus"
            ]):
                claim_verdicts.append({
                    "claim": claim,
                    "entailed": True,
                    "is_hallucination": False,
                    "reasoning": "Correct refusal on an unanswerable case with no supporting source context."
                })
                continue
            verdict = self.verify_single_claim(claim, source_context)
            claim_verdicts.append(verdict)
            if verdict["is_hallucination"]:
                hallucination_count += 1

        is_hallucination = hallucination_count > 0
        total_claims = max(len(claim_verdicts), 1)
        hallucination_ratio = hallucination_count / total_claims

        if is_hallucination:
            failed_reasons = [
                f"Claim '{v['claim'][:40]}...': {v['reasoning']}"
                for v in claim_verdicts if v["is_hallucination"]
            ]
            summary_reasoning = (
                f"Detected {hallucination_count}/{total_claims} hallucinated claims "
                f"({hallucination_ratio:.1%}). Root cause: " + "; ".join(failed_reasons[:2])
            )
        else:
            summary_reasoning = f"All {total_claims} asserted claims fully entailed by source context."

        return is_hallucination, summary_reasoning, claim_verdicts


def detect_hallucinations(
    output_claims: List[str],
    source_context: str,
    llm_client: LLMClient = None
) -> Tuple[bool, str]:
    """
    Convenience wrapper returning binary hallucination flag and reasoning string.
    """
    detector = HallucinationDetector(llm_client=llm_client)
    is_hallucinated, reasoning, _ = detector.detect_hallucinations(output_claims, source_context)
    return is_hallucinated, reasoning
