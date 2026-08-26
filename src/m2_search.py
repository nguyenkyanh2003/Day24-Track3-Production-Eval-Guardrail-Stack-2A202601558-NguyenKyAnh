from __future__ import annotations

"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os
import re
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    BM25_TOP_K,
    COLLECTION_NAME,
    DENSE_TOP_K,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    HYBRID_TOP_K,
    QDRANT_HOST,
    QDRANT_PORT,
)

UPSERT_BATCH_SIZE = 64

_WORD_RE = re.compile(r"\w+", re.UNICODE)

# bge-m3 chiếm ~2.3GB RAM. main.py dựng DenseSearch hai lần (naive baseline rồi
# production) nên cache theo tên model để chỉ nạp một bản duy nhất.
_ENCODER_CACHE: dict[str, object] = {}


def _load_encoder(model_name: str):
    if model_name not in _ENCODER_CACHE:
        from sentence_transformers import SentenceTransformer

        _ENCODER_CACHE[model_name] = SentenceTransformer(model_name)
    return _ENCODER_CACHE[model_name]


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    if not text or not text.strip():
        return text
    try:
        from underthesea import word_tokenize

        # underthesea nối từ ghép bằng "_" ("nghỉ_phép"). BM25 tokenize bằng khoảng
        # trắng, nên query "nghỉ phép" (2 token) sẽ KHÔNG khớp "nghỉ_phép" (1 token).
        # Trả lại khoảng trắng để hai bên cùng hệ token.
        return word_tokenize(text, format="text").replace("_", " ")
    except Exception as e:  # noqa: BLE001 - segmentation lỗi thì vẫn phải search được
        print(f"  ⚠️  Vietnamese segmentation failed: {e}")
        return text


def _tokenize(text: str) -> list[str]:
    """Segment → lowercase → bỏ dấu câu.

    Dùng chung cho cả index lẫn query để corpus và query luôn cùng một hệ token
    (BM25Okapi so khớp token thô, chênh hoa/thường là trượt).
    """
    return _WORD_RE.findall(segment_vietnamese(text).lower())


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        from rank_bm25 import BM25Okapi

        self.documents = chunks
        self.corpus_tokens = [_tokenize(chunk["text"]) for chunk in chunks]
        # BM25Okapi chia cho avgdl → corpus rỗng sẽ ZeroDivisionError.
        self.bm25 = BM25Okapi(self.corpus_tokens) if any(self.corpus_tokens) else None

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if self.bm25 is None:
            return []

        scores = self.bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            SearchResult(
                text=self.documents[i]["text"],
                score=float(scores[i]),
                metadata=self.documents[i].get("metadata", {}),
                method="bm25",
            )
            for i in ranked
            if scores[i] > 0  # score 0 = không chung token nào → nhiễu thuần tuý
        ]


class DenseSearch:
    def __init__(self):
        from qdrant_client import QdrantClient
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            self._encoder = _load_encoder(EMBEDDING_MODEL)
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        from qdrant_client.models import Distance, PointStruct, VectorParams

        if not chunks:
            return

        # Index lại từ đầu mỗi lần chạy: chunking/enrichment đổi thì vector cũ thành rác.
        if self.client.collection_exists(collection):
            self.client.delete_collection(collection)
        self.client.create_collection(
            collection,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )

        vectors = self._get_encoder().encode([c["text"] for c in chunks], show_progress_bar=True)
        points = [
            PointStruct(
                id=i,
                vector=vector.tolist(),
                payload={**chunk.get("metadata", {}), "text": chunk["text"]},
            )
            for i, (chunk, vector) in enumerate(zip(chunks, vectors))
        ]
        for start in range(0, len(points), UPSERT_BATCH_SIZE):
            self.client.upsert(collection, points[start:start + UPSERT_BATCH_SIZE])

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        # qdrant-client >= 1.10 dùng query_points(), search() đã deprecated.
        try:
            response = self.client.query_points(
                collection,
                query=self._get_encoder().encode(query).tolist(),
                limit=top_k,
            )
        except Exception as e:  # noqa: BLE001 - Qdrant chết thì hybrid vẫn chạy bằng BM25
            print(f"  ⚠️  Dense search failed: {e}")
            return []

        return [
            SearchResult(
                text=point.payload.get("text", ""),
                score=float(point.score),
                metadata=point.payload,
                method="dense",
            )
            for point in response.points
        ]


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank).

    RRF chỉ dùng THỨ HẠNG, không dùng score gốc — nhờ vậy không cần normalize giữa
    BM25 (score không chặn trên) và cosine (0..1). Hằng số k làm phẳng đường cong,
    giảm ảnh hưởng của các vị trí đầu bảng.
    """
    fused: dict[str, dict] = {}
    for results in results_list:
        for rank, result in enumerate(results):
            entry = fused.setdefault(result.text, {"score": 0.0, "result": result})
            entry["score"] += 1.0 / (k + rank + 1)

    ranked = sorted(fused.values(), key=lambda entry: entry["score"], reverse=True)[:top_k]
    return [
        SearchResult(
            text=entry["result"].text,
            score=entry["score"],
            metadata=entry["result"].metadata,
            method="hybrid",
        )
        for entry in ranked
    ]


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print("Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
