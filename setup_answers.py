"""
Setup script: chạy Day 18 pipeline trên 50 câu hỏi → lưu answers_50q.json

Chạy TRƯỚC khi bắt đầu Phase A:
    python setup_answers.py

Yêu cầu:
    1. Đã copy src/ từ Day 18 (m1-m5, pipeline.py) vào thư mục này
    2. docker compose up -d  (Qdrant đang chạy trên port 6333)
    3. .env có OPENAI_API_KEY
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def check_day18_files() -> bool:
    required = [
        "src/m1_chunking.py", "src/m2_search.py", "src/m3_rerank.py",
        "src/m4_eval.py",     "src/m5_enrichment.py", "src/pipeline.py",
    ]
    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        print("\n❌ Thiếu files từ Day 18. Copy chúng vào src/ trước:\n")
        for f in missing:
            print(f"   cp <Day18>/src/{os.path.basename(f)} src/")
        return False
    print(f"✓ Day 18 source files: {len(required)}/{len(required)} found")
    return True


def build_pipeline():
    from src.m1_chunking import load_documents, chunk_hierarchical
    from src.m2_search import HybridSearch
    from src.m3_rerank import CrossEncoderReranker
    from src.m5_enrichment import enrich_chunks
    from config import RERANK_TOP_K

    print("\n[1/3] Chunking + enriching documents...")
    t0 = time.time()
    docs = load_documents()
    all_chunks = []
    for doc in docs:
        parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        for child in children:
            all_chunks.append({
                "text": child.text,
                "metadata": {**child.metadata, "parent_id": child.parent_id},
            })

    # Keep setup deterministic and inexpensive. Source/parent metadata already
    # carries the context needed by retrieval; LLM enrichment would add one API
    # request per chunk before the actual 50-question evaluation.
    enriched = enrich_chunks(all_chunks, methods=[])
    if enriched:
        all_chunks = [{"text": e.enriched_text, "metadata": e.auto_metadata} for e in enriched]
        print(f"  ✓ Enriched {len(enriched)} chunks ({time.time()-t0:.1f}s)")
    else:
        print(f"  ✓ Using {len(all_chunks)} raw chunks (M5 not implemented or no API key)")

    print("\n[2/3] Indexing (BM25 + Dense)...")
    t0 = time.time()
    search = HybridSearch()
    search.index(all_chunks)
    print(f"  ✓ Indexed {len(all_chunks)} chunks ({time.time()-t0:.1f}s)")

    print("\n[3/3] Loading reranker...")
    t0 = time.time()
    reranker = CrossEncoderReranker()
    print(f"  ✓ Reranker ready ({time.time()-t0:.1f}s)")

    return search, reranker, RERANK_TOP_K


def run_query(q: str, search, reranker, top_k: int) -> tuple[str, list[str]]:
    from config import (LLM_API_KEY, LLM_MODEL, OPENAI_BASE_URL,
                        RERANK_CANDIDATE_K)

    results = search.search(q)
    docs = [
        {"text": result.text, "score": result.score, "metadata": result.metadata}
        for result in results[:RERANK_CANDIDATE_K]
    ]
    reranked = reranker.rerank(q, docs, top_k=top_k)
    selected = reranked if reranked else results[:3]
    contexts = []
    seen = set()
    for result in selected:
        metadata = result.metadata
        context = metadata.get("parent_text") or result.text
        source = metadata.get("source", "")
        context = f"[Nguồn: {source}]\n{context}" if source else context
        if context not in seen:
            seen.add(context)
            contexts.append(context)

    if LLM_API_KEY and contexts:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=LLM_API_KEY, base_url=OPENAI_BASE_URL or None)
            ctx = "\n\n".join(contexts)
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": (
                        "Trả lời bằng tiếng Việt, CHỈ dựa trên context. Ưu tiên chính sách "
                        "mới/hiện hành khi có xung đột phiên bản. Với câu hỏi có-không, "
                        "trả lời trực tiếp trước khi giải thích. Với câu tính toán, nêu phép "
                        "tính. Nếu context không đủ, nói 'Không tìm thấy.' Tối đa 4 câu."
                    )},
                    {"role": "user",   "content": f"Context:\n{ctx}\n\nCâu hỏi: {q}"},
                ],
            )
            return resp.choices[0].message.content, contexts
        except Exception as e:
            print(f"  ⚠️  LLM generation failed: {e}")

    return (contexts[0] if contexts else "Không tìm thấy thông tin."), contexts


def main():
    print("=" * 60)
    print("LAB 24 SETUP — Generating answers for 50 questions")
    print("=" * 60)

    if not check_day18_files():
        sys.exit(1)

    with open("test_set_50q.json", encoding="utf-8") as f:
        test_set = json.load(f)
    print(f"✓ Loaded {len(test_set)} questions (factual/multi_hop/adversarial)")

    try:
        search, reranker, top_k = build_pipeline()
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("→ Đảm bảo bạn đã copy src/ từ Day 18 và đã pip install -r requirements.txt")
        sys.exit(1)

    print(f"\nRunning {len(test_set)} queries...")
    answers = []
    t_start = time.time()

    for i, item in enumerate(test_set):
        answer, contexts = run_query(item["question"], search, reranker, top_k)
        answers.append({
            "id":           item["id"],
            "distribution": item["distribution"],
            "question":     item["question"],
            "answer":       answer,
            "contexts":     contexts,
            "ground_truth": item["ground_truth"],
        })
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(test_set)}] done ({time.time()-t_start:.0f}s elapsed)")

    with open("answers_50q.json", "w", encoding="utf-8") as f:
        json.dump(answers, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Saved {len(answers)} answers → answers_50q.json")
    print(f"  Total time: {time.time()-t_start:.1f}s")
    print("\n→ Bây giờ bắt đầu Phase A:")
    print("     python src/phase_a_ragas.py")


if __name__ == "__main__":
    main()
