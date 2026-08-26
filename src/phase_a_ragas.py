from __future__ import annotations

"""Phase A: RAGAS Production Evaluation — 50q, 3 distributions, cluster analysis."""

import json
import math
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH, ANSWERS_PATH

Distribution = str  # "factual" | "multi_hop" | "adversarial"

DIAGNOSTIC_TREE = {
    "faithfulness":      ("LLM hallucinating", "Tighten system prompt, lower temperature"),
    "context_recall":    ("Missing relevant chunks", "Improve chunking or add BM25"),
    "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
    "answer_relevancy":  ("Answer doesn't match question", "Improve prompt template"),
}


@dataclass
class RagasResult:
    question_id: int
    distribution: Distribution
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float

    @property
    def avg_score(self) -> float:
        return (self.faithfulness + self.answer_relevancy +
                self.context_precision + self.context_recall) / 4

    @property
    def worst_metric(self) -> str:
        scores = {
            "faithfulness":      self.faithfulness,
            "answer_relevancy":  self.answer_relevancy,
            "context_precision": self.context_precision,
            "context_recall":    self.context_recall,
        }
        return min(scores, key=scores.get)


# ─── Đã implement sẵn ────────────────────────────────────────────────────────

def load_test_set_50q(path: str = TEST_SET_PATH) -> list[dict]:
    """Load 50q test set với 3 distributions."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_answers(path: str = ANSWERS_PATH) -> list[dict]:
    """Load pre-generated answers từ setup_answers.py."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"answers_50q.json không tìm thấy tại {path}\n"
            "→ Chạy trước: python setup_answers.py"
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_phase_a_report(results: list[RagasResult], clusters: dict,
                         path: str = "reports/ragas_50q.json") -> None:
    """Save Phase A report to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    per_dist: dict[str, dict] = {}
    for dist in ["factual", "multi_hop", "adversarial"]:
        subset = [r for r in results if r.distribution == dist]
        if subset:
            per_dist[dist] = {
                "count": len(subset),
                "faithfulness":      sum(r.faithfulness for r in subset) / len(subset),
                "answer_relevancy":  sum(r.answer_relevancy for r in subset) / len(subset),
                "context_precision": sum(r.context_precision for r in subset) / len(subset),
                "context_recall":    sum(r.context_recall for r in subset) / len(subset),
                "avg_score":         sum(r.avg_score for r in subset) / len(subset),
            }

    report = {
        "total_questions": len(results),
        "per_distribution": per_dist,
        "failure_clusters": clusters,
        "overall_avg_score": round(
            sum(result.avg_score for result in results) / len(results), 4
        ) if results else 0.0,
        "bottom_10": bottom_10(results),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Phase A report saved → {path}")


# ─── Tasks 1-4: Sinh viên implement ──────────────────────────────────────────

def group_by_distribution(test_set: list[dict]) -> dict[str, list[dict]]:
    """Task 1: Nhóm 50 câu hỏi theo 3 distributions.

    Returns:
        {"factual": [...], "multi_hop": [...], "adversarial": [...]}
    """
    groups = {"factual": [], "multi_hop": [], "adversarial": []}
    for item in test_set:
        distribution = item.get("distribution")
        if distribution not in groups:
            raise ValueError(f"Unknown distribution: {distribution!r}")
        groups[distribution].append(item)
    return groups


def run_ragas_50q(answers: list[dict]) -> list[RagasResult]:
    """Task 2: Chạy RAGAS 4 metrics trên toàn bộ 50 câu hỏi.

    Gợi ý — import từ Day 18 của bạn:
        from src.m4_eval import evaluate_ragas

    Steps:
        1. Extract questions, answers, contexts, ground_truths từ answers list
        2. Gọi evaluate_ragas() từ m4_eval.py
        3. Kết hợp kết quả với distribution info từ answers list
        4. Return list[RagasResult]
    """
    if not answers:
        return []

    required = {"id", "distribution", "question", "answer", "contexts", "ground_truth"}
    for index, answer in enumerate(answers):
        missing = required - answer.keys()
        if missing:
            raise ValueError(f"Answer #{index} is missing fields: {sorted(missing)}")

    from src.m4_eval import evaluate_ragas

    raw = evaluate_ragas(
        [a["question"] for a in answers],
        [a["answer"] for a in answers],
        [a["contexts"] for a in answers],
        [a["ground_truth"] for a in answers],
    )
    per_question = raw.get("per_question", [])
    if len(per_question) != len(answers):
        raise RuntimeError(
            f"RAGAS returned {len(per_question)} per-question results for "
            f"{len(answers)} answers"
        )

    def score(item, name: str) -> float:
        value = item.get(name) if isinstance(item, dict) else getattr(item, name)
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return 0.0
        return 0.0 if math.isnan(numeric) else max(0.0, min(1.0, numeric))

    return [
        RagasResult(
            question_id=answer["id"], distribution=answer["distribution"],
            question=answer["question"], answer=answer["answer"],
            contexts=answer["contexts"], ground_truth=answer["ground_truth"],
            faithfulness=score(evaluation, "faithfulness"),
            answer_relevancy=score(evaluation, "answer_relevancy"),
            context_precision=score(evaluation, "context_precision"),
            context_recall=score(evaluation, "context_recall"),
        )
        for answer, evaluation in zip(answers, per_question)
    ]


def bottom_10(results: list[RagasResult]) -> list[dict]:
    """Task 3: Lấy 10 câu hỏi có avg_score thấp nhất.

    Returns:
        [{"rank": 1, "question_id": ..., "distribution": ...,
          "question": ..., "avg_score": ..., "worst_metric": ...,
          "diagnosis": ..., "suggested_fix": ...}, ...]
    """
    output = []
    for rank, result in enumerate(sorted(results, key=lambda item: item.avg_score)[:10], 1):
        diagnosis, suggested_fix = DIAGNOSTIC_TREE[result.worst_metric]
        output.append({
            "rank": rank, "question_id": result.question_id,
            "distribution": result.distribution, "question": result.question,
            "avg_score": round(result.avg_score, 4),
            "worst_metric": result.worst_metric, "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
        })
    return output


def cluster_analysis(results: list[RagasResult]) -> dict:
    """Task 4: Phân tích failure clusters theo (worst_metric × distribution).

    Mục tiêu: tìm ra distribution nào hay bị failure nhất và metric nào yếu nhất.

    Returns:
        {
          "matrix": {
            "faithfulness":      {"factual": 3, "multi_hop": 5, "adversarial": 2},
            "answer_relevancy":  {...},
            "context_precision": {...},
            "context_recall":    {...},
          },
          "dominant_failure_distribution": "multi_hop",
          "dominant_failure_metric": "context_recall",
          "insight": "..."
        }
    """
    distributions = ("factual", "multi_hop", "adversarial")
    matrix = {
        metric: {distribution: 0 for distribution in distributions}
        for metric in DIAGNOSTIC_TREE
    }
    for result in results:
        if result.distribution not in distributions:
            raise ValueError(f"Unknown distribution: {result.distribution!r}")
        matrix[result.worst_metric][result.distribution] += 1

    # Every question contributes exactly one cell, so summing the matrix by
    # distribution would merely reproduce the 20/20/10 dataset sizes. The
    # distribution with the lowest mean score is the meaningful failure signal.
    distribution_scores = {
        distribution: [
            result.avg_score for result in results
            if result.distribution == distribution
        ]
        for distribution in distributions
    }
    populated = {
        distribution: scores
        for distribution, scores in distribution_scores.items() if scores
    }
    dominant_distribution = min(
        populated,
        key=lambda distribution: sum(populated[distribution]) / len(populated[distribution]),
    ) if populated else "factual"
    dominant_metric = max(matrix, key=lambda metric: sum(matrix[metric].values()))
    insight = (
        f"Distribution '{dominant_distribution}' có nhiều failure nhất; "
        f"'{dominant_metric}' là metric yếu chủ đạo. "
        f"Khuyến nghị ưu tiên: {DIAGNOSTIC_TREE[dominant_metric][1]}."
    )
    return {
        "matrix": matrix,
        "dominant_failure_distribution": dominant_distribution,
        "dominant_failure_metric": dominant_metric,
        "insight": insight,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_set = load_test_set_50q()
    print(f"Loaded {len(test_set)} questions")

    groups = group_by_distribution(test_set)
    for dist, qs in groups.items():
        print(f"  {dist}: {len(qs)} questions")

    answers = load_answers()
    results = run_ragas_50q(answers)

    if results:
        b10 = bottom_10(results)
        clusters = cluster_analysis(results)
        save_phase_a_report(results, clusters)
        print("\nBottom 10 worst questions:")
        for item in b10:
            print(f"  #{item['rank']} [{item['distribution']}] {item['question'][:50]}... "
                  f"avg={item['avg_score']:.3f} worst={item['worst_metric']}")
        print(f"\nDominant failure: {clusters.get('dominant_failure_distribution')} / "
              f"{clusters.get('dominant_failure_metric')}")
    else:
        print("⚠️  No results — implement run_ragas_50q() first.")
