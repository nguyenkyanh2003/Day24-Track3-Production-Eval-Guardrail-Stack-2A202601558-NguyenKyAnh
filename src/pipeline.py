from __future__ import annotations

"""Production RAG Pipeline — Bài tập NHÓM: ghép M1+M2+M3+M4."""

import json
import os
import sys
import time
from collections import defaultdict
from contextlib import contextmanager

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (LLM_API_KEY, LLM_MODEL, OPENAI_BASE_URL,
                    RERANK_CANDIDATE_K, RERANK_TOP_K)
from src.m1_chunking import chunk_hierarchical, load_documents
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import evaluate_ragas, failure_analysis, load_test_set, save_report
from src.m5_enrichment import enrich_chunks, wait_for_rate_limit

ANSWER_MODEL = LLM_MODEL
LATENCY_REPORT_PATH = os.path.join("reports", "latency_report.json")

# Test set gồm cả câu version-conflict (v2023 vs v2024), câu phủ định và câu tính toán,
# nên prompt phải nói rõ cách xử lý từng loại thay vì chỉ "trả lời dựa trên context".
ANSWER_SYSTEM_PROMPT = """Bạn trả lời câu hỏi chính sách nội bộ, CHỈ dựa trên context được cung cấp.

Quy tắc:
- Mọi câu trong câu trả lời phải suy ra trực tiếp được từ context. Không dùng kiến thức ngoài,
  không suy đoán, không thêm chi tiết context không nói.
- Context có nhiều phiên bản chính sách (v2023/v2024, v1.0/v2.0): chọn phiên bản có ngày hiệu lực
  mới hơn, hoặc phiên bản mà context ghi rõ là "thay thế" phiên bản kia. Dẫn lại đúng số hiệu
  phiên bản như context ghi, và chỉ gọi là "hiện hành" khi context có căn cứ (ngày hiệu lực,
  câu "thay thế", nhãn "phiên bản hiện hành").
- Câu hỏi có/không: trả lời CÓ hoặc KHÔNG ngay câu đầu tiên, rồi mới giải thích.
- Câu hỏi tính toán: nêu các số lấy từ context rồi trình bày phép tính ra kết quả.
- Không tìm thấy trong context: trả lời đúng một câu "Không tìm thấy."

Trả lời bằng tiếng Việt, tối đa 4 câu, đi thẳng vào câu hỏi."""

_latency_ms: dict[str, list[float]] = defaultdict(list)
_llm_client = None


@contextmanager
def _timed(stage: str):
    """Đo thời gian một bước và tích luỹ vào bảng latency (bonus: latency breakdown)."""
    start = time.perf_counter()
    try:
        yield
    finally:
        _latency_ms[stage].append((time.perf_counter() - start) * 1000)


def _get_llm_client():
    """Lazy singleton — 20 câu hỏi dùng chung một HTTP client."""
    global _llm_client
    if _llm_client is None:
        from openai import OpenAI

        _llm_client = OpenAI(api_key=LLM_API_KEY, base_url=OPENAI_BASE_URL or None, max_retries=5)
    return _llm_client


def build_pipeline():
    """Build production RAG pipeline."""
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("=" * 60, flush=True)

    # Step 1: Load & Chunk (M1)
    t0 = time.time()
    print("\n[1/4] Chunking documents...", flush=True)
    with _timed("1_chunking"):
        docs = load_documents()
        all_chunks = []
        for doc in docs:
            # Chỉ index children; parent đi kèm sẵn trong metadata (parent_text) để mở rộng sau.
            _parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
            for child in children:
                all_chunks.append({"text": child.text, "metadata": {**child.metadata, "parent_id": child.parent_id}})
    print(f"  ✓ {len(all_chunks)} chunks from {len(docs)} documents ({time.time()-t0:.1f}s)", flush=True)

    # Step 2: Enrichment (M5)
    t0 = time.time()
    print(f"\n[2/4] Enriching {len(all_chunks)} chunks (M5, 1 API call/chunk)...", flush=True)
    with _timed("2_enrichment"):
        enriched = enrich_chunks(all_chunks)
    if enriched:
        all_chunks = [{"text": e.enriched_text, "metadata": e.auto_metadata} for e in enriched]
        print(f"  ✓ Enriched {len(enriched)} chunks ({time.time()-t0:.1f}s)", flush=True)
    else:
        print("  ⚠️  M5 not implemented — using raw chunks", flush=True)

    # Step 3: Index (M2)
    t0 = time.time()
    print(f"\n[3/4] Indexing {len(all_chunks)} chunks (BM25 + Dense)...", flush=True)
    with _timed("3_indexing"):
        search = HybridSearch()
        search.index(all_chunks)
    print(f"  ✓ Indexed ({time.time()-t0:.1f}s)", flush=True)

    # Step 4: Reranker (M3)
    t0 = time.time()
    print("\n[4/4] Loading reranker...", flush=True)
    with _timed("4_reranker_load"):
        reranker = CrossEncoderReranker()
        reranker._load_model()  # tải trước để latency query không gánh thời gian load model
    print(f"  ✓ Reranker ready ({time.time()-t0:.1f}s)", flush=True)

    return search, reranker


def _expand_to_parents(results) -> list[str]:
    """Retrieve child (precision) → return parent (context) — M1 hierarchical chunking.

    Child ~256 ký tự đủ nhỏ để match chính xác nhưng thường cụt ý; trả nguyên parent cho
    LLM để không mất điều kiện/ngoại lệ nằm ở câu bên cạnh. Nhiều child cùng một parent
    thì gộp làm một context.

    Nhãn [Nguồn: ...] giúp LLM phân biệt bản v2023 với v2024 khi hai bản cùng lọt top.
    """
    contexts, seen = [], set()
    for result in results:
        text = result.metadata.get("parent_text") or result.text
        if text in seen:
            continue
        seen.add(text)
        source = result.metadata.get("source", "")
        contexts.append(f"[Nguồn: {source}]\n{text}" if source else text)
    return contexts


def _generate_answer(query: str, contexts: list[str]) -> str:
    """Sinh câu trả lời từ context. Không có API key → trả context đầu tiên làm fallback."""
    if not contexts:
        return "Không tìm thấy thông tin."
    if not LLM_API_KEY:
        return contexts[0]

    wait_for_rate_limit()
    try:
        response = _get_llm_client().chat.completions.create(
            model=ANSWER_MODEL,
            temperature=0,  # faithfulness: bớt sáng tạo, bám sát context
            messages=[
                {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{chr(10).join(contexts)}\n\nCâu hỏi: {query}"},
            ],
        )
        return response.choices[0].message.content
    except Exception as e:  # noqa: BLE001 - 1 câu lỗi không được làm sập cả vòng eval
        print(f"  ⚠️  LLM generation failed: {e}", flush=True)
        return contexts[0]


def run_query(query: str, search: HybridSearch, reranker: CrossEncoderReranker) -> tuple[str, list[str]]:
    """Run single query through pipeline: hybrid search → rerank → parent expansion → LLM."""
    with _timed("5_search"):
        results = search.search(query)

    with _timed("6_rerank"):
        docs = [
            {"text": result.text, "score": result.score, "metadata": result.metadata}
            for result in results[:RERANK_CANDIDATE_K]
        ]
        reranked = reranker.rerank(query, docs, top_k=RERANK_TOP_K)

    contexts = _expand_to_parents(reranked or results[:RERANK_TOP_K])

    with _timed("7_generation"):
        answer = _generate_answer(query, contexts)

    return answer, contexts


def latency_breakdown() -> dict:
    """Tổng hợp thời gian từng bước — số liệu cho bảng latency trong báo cáo."""
    return {
        stage: {
            "calls": len(samples),
            "total_ms": round(sum(samples), 1),
            "avg_ms": round(sum(samples) / len(samples), 1),
        }
        for stage, samples in sorted(_latency_ms.items())
    }


def save_latency_report(path: str = LATENCY_REPORT_PATH) -> dict:
    """In bảng latency và ghi ra JSON."""
    breakdown = latency_breakdown()

    print("\n" + "=" * 60)
    print("LATENCY BREAKDOWN")
    print("=" * 60)
    print(f"  {'Stage':<20} {'Calls':>6} {'Total (ms)':>12} {'Avg (ms)':>10}")
    for stage, stats in breakdown.items():
        print(f"  {stage:<20} {stats['calls']:>6} {stats['total_ms']:>12.1f} {stats['avg_ms']:>10.1f}")

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(breakdown, f, ensure_ascii=False, indent=2)
    print(f"Latency report saved to {path}")
    return breakdown


def evaluate_pipeline(search: HybridSearch, reranker: CrossEncoderReranker):
    """Run evaluation on test set."""
    test_set = load_test_set()
    print(f"\n[Eval] Running {len(test_set)} queries...", flush=True)
    questions, answers, all_contexts, ground_truths = [], [], [], []

    for i, item in enumerate(test_set):
        answer, contexts = run_query(item["question"], search, reranker)
        questions.append(item["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{i+1}/{len(test_set)}] {item['question'][:50]}...", flush=True)

    t0 = time.time()
    print(f"\n[Eval] Running RAGAS (4 metrics × {len(test_set)} questions)...", flush=True)
    with _timed("8_ragas_eval"):
        results = evaluate_ragas(questions, answers, all_contexts, ground_truths)
    print(f"  ✓ RAGAS done ({time.time()-t0:.1f}s)", flush=True)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        s = results.get(m, 0)
        print(f"  {'✓' if s >= 0.75 else '✗'} {m}: {s:.4f}")

    failures = failure_analysis(results.get("per_question", []))
    save_report(results, failures)
    save_latency_report()
    return results


if __name__ == "__main__":
    start = time.time()
    search, reranker = build_pipeline()
    evaluate_pipeline(search, reranker)
    print(f"\nTotal: {time.time() - start:.1f}s")
