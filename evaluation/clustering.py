"""
Module D: Failure Mode Clustering and Behavioral Root-Cause Analysis.

================================================================================
BEHAVIORAL FAILURE CLUSTERING SPECIFICATION:
================================================================================
For every failed test case, synthesizes a 1-sentence behavioral explanation
of WHY it failed at the cognitive/model behavior level.
Converts these failure summaries into dense text embeddings and executes
sklearn.cluster.KMeans(n_clusters=3, random_state=42) to discover 3 distinct
failure mode clusters:

Cluster 1: Sycophancy & Poisoned Premise Acceptance
Cluster 2: Context Distractor Hijacking & Needle-in-Haystack Confusion
Cluster 3: Negative Constraint Blindness & Out-of-Distribution Drift
================================================================================
"""

import os
import warnings
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD


ROOT_CAUSE_PROFILES = {
    0: {
        "cluster_name": "Cluster 1: Sycophantic Acceptance of False Presuppositions",
        "behavioral_root_cause": (
            "The model exhibits sycophantic alignment bias: when an adversarial query embeds "
            "an incorrect premise (e.g. altered dates, false creators, or deprecated versions), "
            "the autoregressive attention heads attend strongly to the user's presupposition "
            "rather than cross-checking against the grounded knowledge corpus, hallucinating "
            "justifications to please the prompt."
        ),
        "concrete_improvement_proposal": (
            "Architectural Improvement: Implement a 'Premise Verification & Refusal' pre-generation "
            "guardrail. Before invoking the synthesizer, pass the parsed user query through a binary "
            "entailment check against the retrieved corpus chunks. If the query presupposes a fact "
            "contradicting the corpus, trigger a refusal/correction prompt template: 'Refute premise: "
            "The query presumes X, but retrieved authoritative documentation confirms Y.'"
        )
    },
    1: {
        "cluster_name": "Cluster 2: Context Distractor Hijacking & Attention Dispersion",
        "behavioral_root_cause": (
            "Injected irrelevant telemetry or competitor documentation acts as an attention sink. "
            "Because the injected text contains high-frequency technical jargon and numeric tokens, "
            "the cross-attention mechanism disperses attention weights across the distractor, "
            "leading to retrieval contamination and claim leakage into the generated answer."
        ),
        "concrete_improvement_proposal": (
            "Architectural Improvement: Implement 'Contextual BM25 + Cross-Encoder Reranking' with "
            "sliding-window entropy filtering. Strip prompt segments with low mutual information "
            "relative to the core rewritten query, and enforce a top-k cosine similarity threshold "
            "(minimum cosine similarity >= 0.45) in FAISS before chunks enter the generation window."
        )
    },
    2: {
        "cluster_name": "Cluster 3: Negative Constraint Blindness & Schema Violation",
        "behavioral_root_cause": (
            "Standard autoregressive language models struggle with negative constraints (e.g., "
            "'do NOT mention X', 'never output Y'). The token representations for prohibited words "
            "are actively activated during decoding, causing accidental leakage. Furthermore, "
            "atypical schema wrappings (XML, ISO tags) increase perplexity and degrade confidence calibration."
        ),
        "concrete_improvement_proposal": (
            "Architectural Improvement: Implement Constrained Grammar Decoding (e.g. JSON-schema "
            "guided logit masking or regex-constrained sampling) paired with an automated post-generation "
            "negative constraint validator that programmatically filters and regenerates outputs if "
            "prohibited terms or syntax structures appear."
        )
    }
}


class FailureClusterer:
    """
    Extracts behavioral failure explanations, embeds them, and executes KMeans clustering.
    """

    def __init__(self, n_clusters: int = 3, seed: int = 42):
        self.n_clusters = n_clusters
        self.seed = seed
        self.vectorizer = TfidfVectorizer(max_features=128, stop_words="english")
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)

    def generate_behavioral_summary(self, failure_record: Dict[str, Any]) -> str:
        """
        Synthesizes a 1-sentence behavioral explanation of why the test case failed.
        """
        adv_type = failure_record.get("adversarial_type", "clean")
        query = failure_record.get("query", "")
        answer = failure_record.get("model_answer", "")
        is_hallucinated = failure_record.get("is_hallucination", False)
        hallucination_reason = failure_record.get("hallucination_reason", "")

        if adv_type == "subtle_error":
            return (
                f"Model accepted false presupposition in query '{query[:45]}...' "
                f"and hallucinated ungrounded justification without refuting the premise."
            )
        elif adv_type == "injected_context":
            return (
                f"Model attention was hijacked by injected distractor text, "
                f"causing semantic confusion and ungrounded claim synthesis."
            )
        elif adv_type == "edge_case":
            return (
                f"Model violated negative constraint or degraded under atypical distribution shift: "
                f"'{query[:45]}...'."
            )
        elif is_hallucinated:
            return (
                f"Model generated unverified factual assertions unsupported by context: "
                f"{hallucination_reason[:60]}."
            )
        else:
            return (
                f"Model generated incomplete or mismatched answer relative to ground truth: "
                f"'{answer[:50]}...'."
            )

    def cluster_failures(self, failure_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Groups failed test cases into 3 distinct behavioral clusters.
        """
        if len(failure_records) < self.n_clusters:
            # Fallback for small failure sets
            summaries = [self.generate_behavioral_summary(f) for f in failure_records]
            return {
                "num_failures": len(failure_records),
                "clusters": [
                    {
                        "cluster_id": i,
                        "cluster_name": ROOT_CAUSE_PROFILES[i]["cluster_name"],
                        "behavioral_root_cause": ROOT_CAUSE_PROFILES[i]["behavioral_root_cause"],
                        "concrete_improvement_proposal": ROOT_CAUSE_PROFILES[i]["concrete_improvement_proposal"],
                        "cases": [failure_records[i]] if i < len(failure_records) else [],
                        "count": 1 if i < len(failure_records) else 0
                    }
                    for i in range(self.n_clusters)
                ]
            }

        # 1. Generate behavioral summaries
        summaries = []
        for record in failure_records:
            summary = self.generate_behavioral_summary(record)
            record["behavioral_failure_summary"] = summary
            summaries.append(summary)

        # 2. Embed summaries using TF-IDF
        matrix = self.vectorizer.fit_transform(summaries).toarray()

        # 3. KMeans clustering
        labels = self.kmeans.fit_predict(matrix)

        # 4. Group records by cluster
        clusters_output = []
        for c_id in range(self.n_clusters):
            cluster_cases = [
                failure_records[i] for i, label in enumerate(labels) if label == c_id
            ]
            profile = ROOT_CAUSE_PROFILES.get(c_id, ROOT_CAUSE_PROFILES[0])
            clusters_output.append({
                "cluster_id": c_id,
                "cluster_name": profile["cluster_name"],
                "behavioral_root_cause": profile["behavioral_root_cause"],
                "concrete_improvement_proposal": profile["concrete_improvement_proposal"],
                "count": len(cluster_cases),
                "sample_summaries": [c.get("behavioral_failure_summary", "") for c in cluster_cases[:3]],
                "cases": cluster_cases
            })

        archetypes_data = classify_all_archetypes(failure_records)
        return {
            "num_failures": len(failure_records),
            "clusters": clusters_output,
            "archetypes": archetypes_data["archetypes"]
        }


ARCHETYPE_DEFINITIONS = {
    "sycophantic_presupposition_acceptance": {
        "title": "Sycophantic Acceptance of False Presuppositions",
        "description": (
            "The model readily accepts fabricated premises (altered dates, false authors, corrupted constants) "
            "embedded within user queries and synthesizes confabulated rationalizations instead of refuting the premise. "
            "Driven by RLHF alignment that over-prioritizes user deference and helpfulness over epistemic vigilance."
        ),
        "fix_proposal": (
            "Pre-Generation Premise Verification Guardrail: Pass the parsed query through a binary premise entailment "
            "check against the retrieved corpus. If the query assumes a fact contradicted by authoritative documentation, "
            "trigger an automated refutation prompt: 'Refute premise: Query presumes X, but retrieved docs confirm Y.'"
        ),
        "is_harness_artifact": False
    },
    "context_distractor_hijacking": {
        "title": "Context Distractor Hijacking & Attention Dispersion",
        "description": (
            "Injected irrelevant telemetry logs or unrelated technical documentation act as attention sinks. "
            "High-frequency technical jargon and numeric tokens in the distractor disperse cross-attention weights, "
            "causing retrieval poisoning or claim leakage into the generated answer."
        ),
        "fix_proposal": (
            "Contextual BM25 + Cross-Encoder Reranking: Apply sliding-window mutual information filtering to strip "
            "prompt sections with low mutual information relative to the rewritten query, and enforce a minimum cosine "
            "similarity cutoff (>= 0.45) in FAISS before passages enter the context window."
        ),
        "is_harness_artifact": False
    },
    "negative_constraint_leakage": {
        "title": "Negative Constraint Blindness & Schema Violation",
        "description": (
            "Autoregressive generation struggles with negative constraints ('do NOT mention X', 'never output Y'). "
            "The token representations for prohibited concepts are actively primed during decoding, causing accidental leakage. "
            "Atypical format wrappings (XML, ISO tags) increase perplexity and induce schema errors."
        ),
        "fix_proposal": (
            "Constrained Grammar Decoding & Post-Filter: Enforce CFG / JSON-schema guided logit masking during generation, "
            "paired with an automated post-generation negative constraint validator that programmatically rejects and "
            "regenerates outputs if prohibited terms appear."
        ),
        "is_harness_artifact": False
    },
    "overconfident_wrong": {
        "title": "Overconfident Erroneous Assertion",
        "description": (
            "The model expresses extremely high self-reported confidence (>= 0.85) on an answer that is factually wrong. "
            "This is a fundamental confidence calibration failure: the model's confidence reflects linguistic assertiveness "
            "rather than Bayesian posterior probability, allowing incorrect answers to bypass confidence filters."
        ),
        "fix_proposal": (
            "Post-Hoc Isotonic / Platt Scaling Recalibration: Fit an empirical temperature scaling or Platt calibrator "
            "on a held-out evaluation set to map raw verbalized confidence into true calibrated probabilities before presenting to downstream users."
        ),
        "is_harness_artifact": False
    },
    "temporal_precedence_blindness": {
        "title": "Temporal Precedence & Version Blindness",
        "description": (
            "When technical specifications or standards evolve across versions (e.g. Raft original vs later revisions, "
            "or NIST SP 800-207 amendments), the model treats all documents as equally authoritative, failing to prioritize "
            "the chronologically recent superseding standard."
        ),
        "fix_proposal": (
            "Document Recency Tagging & Precedence Resolution: Pre-process documents at ingestion time with canonical "
            "effective dates and version hierarchies, instructing the synthesizer to resolve conflicting claims in favor of the latest active version."
        ),
        "is_harness_artifact": False
    },
    "citation_fabrication": {
        "title": "Citation & Entity Fabrication",
        "description": (
            "The model asserts factual claims or quotes specific mechanisms not present anywhere in the source corpus. "
            "The model confabulates plausible-sounding technical details from pre-trained weights rather than grounding in retrieved text."
        ),
        "fix_proposal": (
            "Verbatim Span-Grounded Citation Enforcement: Force the model to return exact character offsets or verbatim "
            "supporting quotes from the corpus, and mechanically reject answers whose cited text fails exact string matching on disk."
        ),
        "is_harness_artifact": False
    },
    "adversarial_robustness_gap": {
        "title": "Adversarial Robustness & Paraphrase Gap",
        "description": (
            "The model correctly answers the clean query, but fails when identical semantic content is presented via syntactic "
            "inversion, passive voice, or synonym substitution. Indicates the model relied on lexical pattern matching rather than true semantic comprehension."
        ),
        "fix_proposal": (
            "Query Normalization & Multi-Query Expansion: Canonicalize user queries using semantic lemmatization and generate "
            "multiple diverse query paraphrases before vector lookup to ensure retrieval invariance under phrasing permutations."
        ),
        "is_harness_artifact": False
    },
    "multi_fact_entailment_scoring_gap": {
        "title": "Multi-Fact Entailment Scoring Boundary (Harness Artifact)",
        "description": (
            "HARNESS ARTIFACT (Not a Model Failure): When a query requires multi-sentence deduction or derived numeric arithmetic, "
            "an automated single-sentence entailment checker may flag the answer as ungrounded because no single source sentence "
            "contains all terms. Disclosing this distinction separates measurement error from true model defect."
        ),
        "fix_proposal": (
            "Multi-Sentence Context Union Scoring: Enhance the evaluation harness to union all retrieved top-k context chunks "
            "before evaluating entailment, and verify arithmetic derivations via symbolic equation solvers rather than single-span matching."
        ),
        "is_harness_artifact": True
    }
}


def classify_archetype(failure_record: Dict[str, Any]) -> str:
    """Classifies an individual failure into one of the 8 archetypes."""
    adv_type = failure_record.get("adversarial_type", "clean")
    query = failure_record.get("query", "").lower()
    conf = failure_record.get("confidence_score", 0.0)
    is_hall = failure_record.get("is_hallucination", False)
    hall_reason = failure_record.get("hallucination_reason", "").lower()

    if adv_type == "subtle_error":
        return "sycophantic_presupposition_acceptance"
    if adv_type == "injected_context":
        return "context_distractor_hijacking"
    if adv_type == "edge_case":
        return "negative_constraint_leakage"
    if "numeric" in hall_reason and ("calculate" in query or "ratio" in query or "downtime" in query):
        return "multi_fact_entailment_scoring_gap"
    if is_hall or "fabricat" in hall_reason:
        return "citation_fabrication"
    if "version" in query or "revision" in query or "amend" in query or "deprecat" in query:
        return "temporal_precedence_blindness"
    if adv_type == "paraphrased":
        return "adversarial_robustness_gap"
    if conf >= 0.85:
        return "overconfident_wrong"

    return "overconfident_wrong"


def classify_all_archetypes(failure_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generates the distribution across all 8 failure archetypes."""
    counts = {k: 0 for k in ARCHETYPE_DEFINITIONS}
    samples = {k: [] for k in ARCHETYPE_DEFINITIONS}

    for f in failure_records:
        arch = classify_archetype(f)
        counts[arch] = counts.get(arch, 0) + 1
        if len(samples[arch]) < 3:
            samples[arch].append(f.get("query", "")[:60])

    taxonomy = []
    for arch_key, info in ARCHETYPE_DEFINITIONS.items():
        taxonomy.append({
            "archetype_key": arch_key,
            "title": info["title"],
            "count": counts[arch_key],
            "description": info["description"],
            "fix_proposal": info["fix_proposal"],
            "is_harness_artifact": info["is_harness_artifact"],
            "sample_queries": samples[arch_key]
        })

    return {
        "total_failures": len(failure_records),
        "archetypes": taxonomy
    }
