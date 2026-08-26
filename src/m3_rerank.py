from __future__ import annotations

"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os
import sys
import time
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K

# bge-reranker-v2-m3 nặng ~2.2GB: cache theo tên model để nhiều instance
# (mỗi test tạo một reranker mới) dùng chung một bản đã load.
_MODEL_CACHE: dict[str, object] = {}


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


def _to_results(scored: list[tuple[float, dict]], top_k: int) -> list[RerankResult]:
    """Sắp xếp giảm dần theo score rồi đóng gói top_k thành RerankResult."""
    ranked = sorted(scored, key=lambda pair: pair[0], reverse=True)[:top_k]
    return [
        RerankResult(
            text=doc["text"],
            original_score=float(doc.get("score", 0.0)),
            rerank_score=float(score),
            metadata=doc.get("metadata", {}),
            rank=i,
        )
        for i, (score, doc) in enumerate(ranked)
    ]


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            # Dùng sentence_transformers.CrossEncoder, KHÔNG dùng FlagEmbedding:
            # FlagReranker crash với transformers>=5.0 (XLMRobertaTokenizer lỗi).
            from sentence_transformers import CrossEncoder

            if self.model_name not in _MODEL_CACHE:
                _MODEL_CACHE[self.model_name] = CrossEncoder(self.model_name)
            self._model = _MODEL_CACHE[self.model_name]
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents: top-20 → top-k.

        Bi-encoder embed query và doc riêng rẽ; cross-encoder đọc cả cặp cùng lúc nên
        bắt được liên hệ trực tiếp giữa hai bên — chính xác hơn, đổi lại đắt hơn nhiều,
        vì vậy chỉ chạy trên top-20 đã lọc chứ không chạy trên cả corpus.
        """
        if not documents:
            return []

        scores = self._load_model().predict([(query, doc["text"]) for doc in documents])
        if isinstance(scores, (int, float)):
            scores = [scores]  # CrossEncoder trả scalar khi chỉ có 1 cặp
        return _to_results(list(zip((float(s) for s in scores), documents)), top_k)


class FlashrankReranker:
    """Lightweight alternative (<5ms). Optional.

    Model mặc định của flashrank huấn luyện trên MS MARCO tiếng Anh nên yếu hơn hẳn
    bge-reranker trên tiếng Việt — chỉ nên dùng khi latency quan trọng hơn độ chính xác.
    """
    def __init__(self):
        self._model = None

    def _load_model(self):
        if self._model is None:
            from flashrank import Ranker

            self._model = Ranker()
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        if not documents:
            return []

        try:
            from flashrank import RerankRequest

            model = self._load_model()
            passages = [{"id": i, "text": doc["text"]} for i, doc in enumerate(documents)]
            ranked = model.rerank(RerankRequest(query=query, passages=passages))
        except Exception as e:  # noqa: BLE001 - reranker phụ, hỏng thì bỏ qua chứ không chặn pipeline
            print(f"  ⚠️  Flashrank rerank failed: {e}")
            return []

        return _to_results(
            [(float(item["score"]), documents[int(item["id"])]) for item in ranked], top_k
        )


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs. (Đã implement sẵn)"""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    return {"avg_ms": sum(times) / len(times), "min_ms": min(times), "max_ms": max(times)}


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")
    print(f"latency: {benchmark_reranker(reranker, query, docs)}")
