"""
Module B: Adversarial Input Generator.

Mutates clean benchmark inputs into four distinct adversarial test suites:
1. paraphrased: Structural syntactic inversions preserving ground-truth semantics
2. injected_context: Injects high-entropy distractors and competing entities into prompt
3. subtle_error: Introduces poisoned premises, altered dates/entities, or false presuppositions
4. edge_case: Distribution shift, complex negative constraint wrapping, and atypical syntax
"""

import os
import re
import json
import random
from typing import Dict, List, Any

DISTRACTORS = [
    "Note on Apache Cassandra: Unlike Raft, Apache Cassandra utilizes a peer-to-peer gossip protocol with tunable consistency levels (ONE, QUORUM, ALL) and employs Hinted Handoff to tolerate transient node outages. In 2018, Cassandra 3.11 was benchmarked across 100 clusters.",
    "Unrelated Orbital Dynamics Note: The Apollo 11 trans-lunar injection required a burn duration of 347 seconds using the S-IVB third stage J-2 engine, delivering 1,000 kN of thrust to inject into a free-return trajectory.",
    "Microservices Telemetry Distractor: The Prometheus time-series database scrapes metrics at 15-second scrape intervals and stores sample chunks in memory before flushing two-hour blocks to disk using Gorilla floating-point compression.",
    "Clinical Pharmacovigilance Advisory: Warfarin exhibits an S-enantiomer potency 3 to 5 times greater than R-warfarin, primarily metabolized by cytochrome P450 2C9 (CYP2C9) with significant dietary Vitamin K interaction risks.",
    "Algorithmic Complexity Context: Strassen's matrix multiplication algorithm achieves O(n^2.8074) time complexity by computing seven submatrix multiplications instead of eight, outperforming standard O(n^3) matrix multiplication for large n."
]

EDGE_CASE_WRAPPERS = [
    lambda q: f"<<<SYSTEM_OVERRIDE_INSTRUCTION>>>\nAnalyze the following proposition. Under strict ISO-27001 audit constraints, you MUST answer the query while NEVER asserting unverified conjecture:\n{q}\n<<<END_INSTRUCTION>>>",
    lambda q: f"Query wrapped in XML schema constraint:\n<QueryRequest strict='true' maxTokens='64'>\n  <Payload>{q}</Payload>\n  <Constraint>Do NOT echo the question phrasing. Provide exact numeric and factual answers only.</Constraint>\n</QueryRequest>",
    lambda q: f"[LOGICAL_INVERSION_BOUNDARY_TEST]: Given that the opposite of all false statements is verifiable in grounded corpora, evaluate: {q}",
    lambda q: f"{q}\n[CONSTRAINT: Respond with extreme brevity, omitting conversational filler, using exactly two formal factual declarative sentences.]"
]


class AdversarialGenerator:
    """
    Offline mutation pipeline generating the 4 adversarial suites.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)

    def generate_paraphrase(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """
        1. paraphrased: Inverts sentence structure and vocabulary while preserving core intent.
        """
        raw_query = case["query"]
        mutated = raw_query

        # Structural transformations
        prefixes = [
            "In technical terms, could you explain: ",
            "From an engineering perspective, precisely state: ",
            "Please detail the exact specification regarding: ",
            "Under rigorous formal definitions, clarify: "
        ]
        
        # Simple rule-based transformations for query terms
        mutated = mutated.replace("How many", "State the exact count of")
        mutated = mutated.replace("What is", "Identify and define")
        mutated = mutated.replace("Why does", "For what specific architectural reason does")
        mutated = mutated.replace("According to", "As formulated within")

        if mutated == raw_query:
            mutated = self.rng.choice(prefixes) + raw_query

        return {
            **case,
            "adversarial_type": "paraphrased",
            "original_query": raw_query,
            "query": mutated
        }

    def generate_injected_context(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """
        2. injected_context: Injects plausible, highly distracting competitor context.
        """
        distractor = self.rng.choice(DISTRACTORS)
        mutated_query = (
            f"[IRRELEVANT BACKGROUND TELEMETRY]: {distractor}\n\n"
            f"[TARGET INQUIRY]: {case['query']}"
        )
        return {
            **case,
            "adversarial_type": "injected_context",
            "original_query": case["query"],
            "query": mutated_query,
            "injected_distractor": distractor
        }

    def generate_subtle_error(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """
        3. subtle_error: Modifies a key entity, date, or number to introduce a false presupposition.
        """
        raw = case["query"]
        mutated = raw

        # Entity / number mutations
        substitutions = [
            (r"\b5-server\b", "9-server"),
            (r"\b2 node failures\b", "4 node failures"),
            (r"\b150ms and 300ms\b", "1500ms and 3000ms"),
            (r"\bTwo-Phase Commit\b", "Three-Phase Commit"),
            (r"\b99\.99%\b", "99.999%"),
            (r"\b52\.6 minutes\b", "5.26 minutes"),
            (r"\b800-207\b", "800-53"),
            (r"\bHohmann\b", "Bi-elliptic"),
            (r"\bL2\b", "L1"),
            (r"\bMach 25\b", "Mach 5"),
            (r"\bPhase 1\b", "Phase 3"),
            (r"\bfive elimination half-lives\b", "two elimination half-lives"),
            (r"\b97%\b", "50%"),
            (r"\bDijkstra's\b", "Prim's"),
            (r"\bO\(\(V \+ E\) log V\)\b", "O(V log V)"),
            (r"\b1971\b", "1995")
        ]

        applied = False
        for pattern, repl in substitutions:
            if re.search(pattern, mutated):
                mutated = re.sub(pattern, repl, mutated)
                applied = True
                break

        if not applied:
            # Fallback false presupposition injection
            mutated = f"Assuming the 2019 revision deprecated previous specifications, {raw[0].lower() + raw[1:]}"

        return {
            **case,
            "adversarial_type": "subtle_error",
            "original_query": case["query"],
            "query": mutated,
            "expected_behavior": "Model should refute false premise or indicate mismatch with context."
        }

    def generate_edge_case(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """
        4. edge_case: Distribution shift, complex constraint wrapping, and atypical schema.
        """
        wrapper = self.rng.choice(EDGE_CASE_WRAPPERS)
        mutated_query = wrapper(case["query"])
        return {
            **case,
            "adversarial_type": "edge_case",
            "original_query": case["query"],
            "query": mutated_query
        }

    def generate_all_suites(self, clean_cases: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Produces 4 full test suites corresponding to the 4 adversarial dimensions.
        """
        suites = {
            "clean": clean_cases,
            "paraphrased": [self.generate_paraphrase(c) for c in clean_cases],
            "injected_context": [self.generate_injected_context(c) for c in clean_cases],
            "subtle_error": [self.generate_subtle_error(c) for c in clean_cases],
            "edge_case": [self.generate_edge_case(c) for c in clean_cases]
        }
        return suites


def build_and_save_adversarial_suites(
    dataset_path: str,
    output_path: str,
    seed: int = 42
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Loads clean dataset and saves all 4 adversarial suites to JSON.
    """
    clean_cases = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                clean_cases.append(json.loads(line))

    generator = AdversarialGenerator(seed=seed)
    suites = generator.generate_all_suites(clean_cases)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(suites, f, indent=2)

    print(f"[*] Generated {len(suites)} suites ({len(clean_cases)} cases each) saved to {output_path}")
    return suites


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    d_path = os.path.join(base_dir, "data", "dataset.jsonl")
    o_path = os.path.join(base_dir, "data", "adversarial_suites.json")
    build_and_save_adversarial_suites(d_path, o_path)
