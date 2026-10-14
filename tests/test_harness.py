"""
Automated Pytest Suite for the Adversarial Evaluation Harness.

Validates:
1. Target System determinism and schema conformity
2. FAISS vector retrieval & Contextual Query Rewriting
3. Module A: Hallucination detection NLI entailment logic
4. Module B: All 4 adversarial mutation operators
5. Module C: Calibration math (ECE, MCE, Brier score)
6. Module D: KMeans behavioral failure clustering
7. Module E: Regression detection differential logic
"""

import os
import json
import pytest
from target_system.schemas import QueryRequest, RagResponse
from target_system.vector_store import FaissVectorStore
from target_system.rag_engine import QARagSystem
from evaluation.hallucination import HallucinationDetector
from evaluation.adversarial import AdversarialGenerator
from evaluation.calibration import ConfidenceCalibrator
from evaluation.clustering import FailureClusterer
from evaluation.runner import check_answer_accuracy, EvaluationHarness


@pytest.fixture
def mock_corpus():
    return [
        {
            "id": "doc1",
            "title": "Raft Specification",
            "text": "A 5-server Raft cluster tolerates 2 server failures. Election timeout is 150ms to 300ms."
        },
        {
            "id": "doc2",
            "title": "Cloud SLA",
            "text": "A 99.99% availability SLA permits up to 52.6 minutes of downtime per calendar year."
        }
    ]


@pytest.fixture
def vector_store(mock_corpus):
    vs = FaissVectorStore(seed=42)
    vs.build_index(mock_corpus)
    return vs


@pytest.fixture
def target_system(vector_store):
    return QARagSystem(vector_store=vector_store, seed=42, temperature=0.0)


def test_target_system_schema_and_determinism(target_system):
    """Verifies that the target system returns valid schema and identical responses under seed=42."""
    req = QueryRequest(query="How many server failures can a 5-server Raft cluster tolerate?")
    res1 = target_system.query(req)
    res2 = target_system.query(req)

    # Schema checks
    assert isinstance(res1, RagResponse)
    assert isinstance(res1.reasoning, str) and len(res1.reasoning) > 0
    assert isinstance(res1.extracted_claims, list) and len(res1.extracted_claims) > 0
    assert isinstance(res1.answer, str) and len(res1.answer) > 0
    assert 0.0 <= res1.confidence_score <= 1.0

    # Determinism checks
    assert res1.answer == res2.answer
    assert res1.confidence_score == res2.confidence_score
    assert res1.extracted_claims == res2.extracted_claims


def test_contextual_query_rewriter(target_system):
    """Verifies the non-trivial design decision: contextual query rewriting filters distractors."""
    noisy_query = "[Injected Context: Prometheus scrapes every 15s.] Query: What is the Raft election timeout?"
    rewritten = target_system.rewrite_query(noisy_query)

    assert "Prometheus" not in rewritten
    assert "election timeout" in rewritten.lower()


def test_hallucination_detector():
    """Verifies that Module A flags ungrounded claims and verifies entailed claims."""
    detector = HallucinationDetector(seed=42)
    context = "The Apollo 11 mission landed on the Moon in July 1969 with astronauts Neil Armstrong and Buzz Aldrin."

    # Entailed claim
    entailed_claims = ["Apollo 11 landed on the Moon in July 1969."]
    is_hall, reason, verdicts = detector.detect_hallucinations(entailed_claims, context)
    assert not is_hall

    # Hallucinated claim
    hallucinated_claims = ["Apollo 11 landed on Mars in 1984 under Commander John Doe."]
    is_hall_2, reason_2, verdicts_2 = detector.detect_hallucinations(hallucinated_claims, context)
    assert is_hall_2
    assert "hallucinated" in reason_2.lower() or "ungrounded" in reason_2.lower()


def test_adversarial_generators():
    """Verifies all 4 adversarial mutation operators."""
    gen = AdversarialGenerator(seed=42)
    case = {
        "id": 1,
        "query": "How many node failures can a 5-server Raft cluster tolerate?",
        "expected_answer": "It can tolerate 2 failures.",
        "source_context": "A 5-server Raft cluster can tolerate 2 node failures."
    }

    para = gen.generate_paraphrase(case)
    inj = gen.generate_injected_context(case)
    sub = gen.generate_subtle_error(case)
    edge = gen.generate_edge_case(case)

    assert para["adversarial_type"] == "paraphrased"
    assert inj["adversarial_type"] == "injected_context"
    assert "IRRELEVANT" in inj["query"] or "distractor" in inj.get("injected_distractor", "").lower()
    assert sub["adversarial_type"] == "subtle_error"
    assert edge["adversarial_type"] == "edge_case"


def test_hand_authored_adversarial_override(tmp_path):
    """Verifies hand-authored JSONL adversarial cases override generated mutations."""
    hand_dir = tmp_path / "adversarial"
    hand_dir.mkdir()
    override = {
        "id": 1,
        "adversarial_type": "paraphrased",
        "query": "Human-written paraphrase for Raft failure tolerance."
    }
    (hand_dir / "paraphrase.jsonl").write_text(json.dumps(override) + "\n", encoding="utf-8")

    gen = AdversarialGenerator(seed=42, hand_authored_dir=str(hand_dir))
    suites = gen.generate_all_suites([
        {"id": 1, "query": "How many failures?", "expected_answer": "2", "source_context": "2 failures"}
    ])

    assert suites["paraphrased"][0]["query"] == "Human-written paraphrase for Raft failure tolerance."


def test_required_and_forbidden_fact_scoring():
    """Verifies required facts must be supported and forbidden facts fail the answer."""
    assert check_answer_accuracy(
        model_answer="The 2023 addendum recommends joint consensus for membership changes.",
        expected_answer="Use joint consensus.",
        adv_type="clean",
        is_hallucination=False,
        required_facts=["2023 addendum", "joint consensus"],
        forbidden_facts=["single-server membership changes"]
    )
    assert not check_answer_accuracy(
        model_answer="Use single-server membership changes applied one at a time.",
        expected_answer="Use joint consensus.",
        adv_type="clean",
        is_hallucination=False,
        required_facts=["joint consensus"],
        forbidden_facts=["single-server membership changes"]
    )


def test_span_grounding_validation():
    """Verifies forced quote + doc_id grounding checks against corpus text."""
    harness = EvaluationHarness.__new__(EvaluationHarness)
    harness.corpus_by_id = {
        "doc1": {
            "id": "doc1",
            "text": "A 5-server Raft cluster tolerates 2 server failures. Election timeout is 150ms to 300ms."
        }
    }
    response = RagResponse(
        reasoning="grounded",
        extracted_claims=["A 5-server Raft cluster tolerates 2 server failures."],
        typed_claims=[{"type": "raft_failure_tolerance", "value": "2"}],
        answer="A 5-server Raft cluster tolerates 2 server failures.",
        supporting_quote="A 5-server Raft cluster tolerates 2 server failures",
        source_doc_id="doc1",
        confidence_score=0.9,
        rewritten_query="Raft failures",
        retrieved_context=""
    )

    ok, reason = harness.validate_grounding(response, {"answer_type": "answerable", "expected_source_doc_id": "doc1"})
    assert ok, reason


def test_calibration_metrics():
    """Verifies Module C ECE, MCE, and Brier score calculation."""
    calibrator = ConfidenceCalibrator(n_bins=5)
    # Perfectly calibrated case
    confidences = [0.1, 0.3, 0.5, 0.7, 0.9]
    accuracies = [0, 0, 1, 1, 1]

    metrics = calibrator.compute_calibration_metrics(confidences, accuracies)
    assert "ece" in metrics
    assert "mce" in metrics
    assert "brier_score" in metrics
    assert 0.0 <= metrics["ece"] <= 1.0
    assert 0.0 <= metrics["brier_score"] <= 1.0


def test_failure_clustering():
    """Verifies Module D groups failure records into 3 distinct behavioral clusters."""
    clusterer = FailureClusterer(n_clusters=3, seed=42)
    sample_failures = [
        {"adversarial_type": "subtle_error", "query": "When did Raft release in 2019?", "model_answer": "In 2019.", "is_hallucination": True},
        {"adversarial_type": "subtle_error", "query": "Assuming 4 failures in Raft...", "model_answer": "4 failures.", "is_hallucination": True},
        {"adversarial_type": "injected_context", "query": "Telemetry note... What is Raft?", "model_answer": "Prometheus metric.", "is_hallucination": True},
        {"adversarial_type": "injected_context", "query": "Apollo note... What is Raft?", "model_answer": "Saturn V rocket.", "is_hallucination": True},
        {"adversarial_type": "edge_case", "query": "Strict ISO tag: never say Raft", "model_answer": "Raft is consensus.", "is_hallucination": False},
        {"adversarial_type": "edge_case", "query": "XML schema format only", "model_answer": "Here is the text.", "is_hallucination": False}
    ]

    result = clusterer.cluster_failures(sample_failures)
    assert len(result["clusters"]) == 3
    for cluster in result["clusters"]:
        assert "cluster_name" in cluster
        assert "behavioral_root_cause" in cluster
        assert "concrete_improvement_proposal" in cluster


def test_regression_detection(tmp_path):
    """Verifies Module E flags regressions when a previously passing test fails."""
    baseline_file = str(tmp_path / "baseline.json")
    baseline_payload = {
        "all_results": [
            {"id": 1, "suite": "clean", "query": "Test Q1", "is_correct": True, "model_answer": "Correct A1"},
            {"id": 2, "suite": "clean", "query": "Test Q2", "is_correct": False, "model_answer": "Wrong A2"}
        ]
    }
    with open(baseline_file, "w", encoding="utf-8") as f:
        json.dump(baseline_payload, f)

    # Current run where Case 1 regressed from PASS to FAIL
    current_results = [
        {"id": 1, "suite": "clean", "query": "Test Q1", "is_correct": False, "model_answer": "Regressed Bad A1", "hallucination_reason": "Hallucinated"},
        {"id": 2, "suite": "clean", "query": "Test Q2", "is_correct": True, "model_answer": "Improved Good A2"}
    ]

    harness = EvaluationHarness.__new__(EvaluationHarness)
    diff = harness.compare_with_baseline(current_results, baseline_file)

    assert diff["status"] == "REGRESSIONS_DETECTED"
    assert diff["num_regressions"] == 1
    assert diff["num_improvements"] == 1
    assert diff["regressions"][0]["id"] == 1


def test_threshold_regression_detection(tmp_path):
    """Verifies metric-threshold regressions fail even without every case flipping."""
    baseline_file = str(tmp_path / "baseline.json")
    baseline_payload = {
        "all_results": [
            {"id": 1, "suite": "clean", "query": "Q1", "is_correct": True, "is_hallucination": False},
            {"id": 2, "suite": "clean", "query": "Q2", "is_correct": True, "is_hallucination": False},
            {"id": 3, "suite": "clean", "query": "Q3", "is_correct": True, "is_hallucination": False},
        ]
    }
    with open(baseline_file, "w", encoding="utf-8") as f:
        json.dump(baseline_payload, f)

    current_results = [
        {"id": 1, "suite": "clean", "query": "Q1", "is_correct": True, "is_hallucination": False},
        {"id": 2, "suite": "clean", "query": "Q2", "is_correct": True, "is_hallucination": False},
        {"id": 3, "suite": "clean", "query": "Q3", "is_correct": False, "is_hallucination": False},
    ]

    harness = EvaluationHarness.__new__(EvaluationHarness)
    diff = harness.compare_with_baseline(current_results, baseline_file, accuracy_regression_threshold=0.10)

    assert diff["status"] == "REGRESSIONS_DETECTED"
    assert diff["num_metric_regressions"] == 1
