from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import json
import math
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    LLM_MAX_RPM,
    LLM_MODEL,
    LLM_API_KEY,
    OPENAI_BASE_URL,
    RAGAS_EMBEDDING_MODEL,
    RAGAS_MAX_RETRIES,
    RAGAS_MAX_WORKERS,
    RAGAS_TIMEOUT,
    TEST_SET_PATH,
)

METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")

# Diagnostic Tree: metric tệ nhất → nguyên nhân gốc → chỗ cần sửa trong pipeline.
DIAGNOSTIC_TREE = {
    "faithfulness": (
        "LLM bịa — câu trả lời chứa thông tin không có trong context",
        "Siết prompt ('chỉ trả lời dựa trên context'), hạ temperature, bắt trích dẫn nguồn",
    ),
    "answer_relevancy": (
        "Câu trả lời lạc đề hoặc lan man so với câu hỏi",
        "Sửa prompt template, thêm query rewriting, yêu cầu trả lời trực tiếp câu hỏi",
    ),
    "context_precision": (
        "Chunk nhiễu lọt vào top-k, chunk đúng bị đẩy xuống dưới",
        "Thêm/siết reranking, lọc theo metadata, giảm top-k",
    ),
    "context_recall": (
        "Retrieval bỏ sót chunk chứa thông tin cần thiết",
        "Chỉnh chunking (parent-child), tăng top-k, thêm BM25 vào hybrid search",
    ),
}


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _as_score(value) -> float:
    """Ép giá trị RAGAS về float sạch. NaN (metric không tính được) → 0.0."""
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if math.isnan(score) else score


def _fan_out_completions(langchain_llm):
    """Bọc LLM lại để n completions thành n request riêng.

    RAGAS coi mọi ChatOpenAI là hỗ trợ n>1 nên answer_relevancy (strictness=3) gửi
    thẳng n=3. Endpoint OpenAI-compatible của Gemini từ chối:
        400 "Multiple candidates is not enabled for this model"
    → RAGAS parse fail → answer_relevancy = NaN. Gửi n prompt giống nhau rồi gộp lại
    cho RAGAS thấy đúng hình dạng n completions, giữ nguyên strictness=3 của metric.
    """
    from ragas.llms import LangchainLLMWrapper

    class FanOutLLM(LangchainLLMWrapper):
        def _fan_out(self, prompt, n, temperature, stop, callbacks) -> dict:
            return {
                "prompts": [prompt] * n,
                "temperature": self.get_temperature(n) if temperature is None else temperature,
                "stop": stop,
                "callbacks": callbacks,
            }

        @staticmethod
        def _as_n_completions(result):
            """n kết quả 1-completion → 1 kết quả n-completion, đúng dạng RAGAS chờ."""
            result.generations = [[generation[0] for generation in result.generations]]
            return result

        def generate_text(self, prompt, n=1, temperature=None, stop=None, callbacks=None):
            return self._as_n_completions(self.langchain_llm.generate_prompt(
                **self._fan_out(prompt, n, temperature, stop, callbacks)))

        async def agenerate_text(self, prompt, n=1, temperature=None, stop=None, callbacks=None):
            return self._as_n_completions(await self.langchain_llm.agenerate_prompt(
                **self._fan_out(prompt, n, temperature, stop, callbacks)))

    return FanOutLLM(langchain_llm)


def _judge_llm():
    """LLM chấm điểm cho RAGAS.

    Trỏ tường minh vào base_url/model trong config thay vì để RAGAS tự lấy default
    OpenAI — nhờ vậy chạy được qua gateway (OpenRouter, Gemini) mà không đổi code.
    """
    from langchain_core.rate_limiters import InMemoryRateLimiter
    from langchain_openai import ChatOpenAI

    judge = ChatOpenAI(
        model=LLM_MODEL,
        api_key=LLM_API_KEY,
        base_url=OPENAI_BASE_URL or None,
        temperature=0,
        max_retries=RAGAS_MAX_RETRIES,
        timeout=RAGAS_TIMEOUT,
        # RAGAS bắn song song nhiều job; không chặn nhịp thì free tier trả 429 hàng loạt
        # và metric rơi về NaN. max_bucket_size=1 → phát đều chứ không cho burst.
        rate_limiter=InMemoryRateLimiter(
            requests_per_second=LLM_MAX_RPM / 60,
            check_every_n_seconds=0.5,
            max_bucket_size=1,
        ) if LLM_MAX_RPM > 0 else None,
    )
    # Chỉ OpenAI gốc mới chắc chắn hỗ trợ n>1; qua gateway thì fan-out cho an toàn.
    return judge if not OPENAI_BASE_URL else _fan_out_completions(judge)


def _judge_embeddings():
    """Embeddings local cho answer_relevancy — gateway không có endpoint embeddings."""
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        # Bản trong langchain_community đã deprecated nhưng vẫn chạy, dùng khi chưa
        # cài gói langchain-huggingface tách riêng.
        from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=RAGAS_EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation.

    4 metrics chia làm hai nhóm:
    - generation: faithfulness (bám context) + answer_relevancy (đúng câu hỏi)
    - retrieval:  context_precision (xếp hạng) + context_recall (độ phủ)

    RAGAS cần OPENAI_API_KEY và Python 3.11+; thiếu thì trả về scores 0 để pipeline
    vẫn chạy hết chứ không sập giữa chừng.
    """
    empty = {metric: 0.0 for metric in METRIC_NAMES} | {"per_question": []}
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
        from ragas.run_config import RunConfig

        dataset = Dataset.from_dict({
            "question": questions, "answer": answers,
            "contexts": contexts, "ground_truth": ground_truths,
        })
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=_judge_llm(),
            embeddings=_judge_embeddings(),
            run_config=RunConfig(max_workers=RAGAS_MAX_WORKERS, max_retries=RAGAS_MAX_RETRIES,
                                 timeout=RAGAS_TIMEOUT),
        )
        df = result.to_pandas()

        per_question = [
            EvalResult(
                question=row["question"],
                answer=row["answer"],
                contexts=list(row["contexts"]),
                ground_truth=row["ground_truth"],
                **{metric: _as_score(row.get(metric)) for metric in METRIC_NAMES},
            )
            for _, row in df.iterrows()
        ]
        aggregate = {
            metric: round(sum(getattr(r, metric) for r in per_question) / len(per_question), 4)
            if per_question else 0.0
            for metric in METRIC_NAMES
        }
        return {**aggregate, "per_question": per_question}
    except Exception as e:  # noqa: BLE001 - eval hỏng không được làm sập pipeline
        print(f"  ⚠️  RAGAS evaluation failed: {e}")
        return empty


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree.

    Xếp hạng câu hỏi theo điểm trung bình 4 metrics, rồi với mỗi câu tệ nhất truy ngược
    metric thấp nhất qua Diagnostic Tree để ra chẩn đoán và hướng sửa.
    """
    failures = []
    for result in eval_results:
        scores = {metric: getattr(result, metric) for metric in METRIC_NAMES}
        worst_metric = min(scores, key=scores.get)
        diagnosis, suggested_fix = DIAGNOSTIC_TREE[worst_metric]
        failures.append({
            "question": result.question,
            "answer": result.answer,
            "ground_truth": result.ground_truth,
            "contexts": result.contexts,
            "scores": {metric: round(score, 4) for metric, score in scores.items()},
            "avg_score": round(sum(scores.values()) / len(scores), 4),
            "worst_metric": worst_metric,
            "score": round(scores[worst_metric], 4),
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
        })

    failures.sort(key=lambda failure: failure["avg_score"])
    return failures[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
