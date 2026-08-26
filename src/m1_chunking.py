from __future__ import annotations

"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import glob
import os
import re
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DATA_DIR,
    HIERARCHICAL_CHILD_SIZE,
    HIERARCHICAL_PARENT_SIZE,
    SEMANTIC_THRESHOLD,
)

# Model nhẹ, đủ dùng để đo độ tương đồng câu-với-câu khi tách chunk.
SEMANTIC_MODEL = "all-MiniLM-L6-v2"

# Không tách chunk semantic khi chunk hiện tại còn quá ngắn — tránh vỡ vụn thành
# các mẩu 1 câu (heading, câu dẫn) vốn vô dụng khi retrieve.
SEMANTIC_MIN_CHARS = 100

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n\n")
_HEADER_RE = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)

_encoder = None


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _extract_pdf_text(path: str) -> str:
    """Extract text layer từ PDF. Trả về "" nếu PDF là scan ảnh (không có text)."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load tất cả markdown và PDF (có text layer) từ data/. (Đã implement sẵn)

    - .md: đọc trực tiếp.
    - .pdf: trích text layer bằng pypdf. PDF scan ảnh (không có text) bị bỏ qua
      kèm cảnh báo — RAG text-based không xử lý được scan nếu chưa OCR.
    """
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(fp)
        if text:
            docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
        else:
            print(f"  ⚠️  Bỏ qua {os.path.basename(fp)}: PDF scan ảnh, không có text layer (cần OCR).")

    return docs


# ─── Helpers dùng chung ──────────────────────────────────


def _split_sentences(text: str) -> list[str]:
    """Tách câu theo dấu kết câu hoặc dòng trống."""
    return [s.strip() for s in _SENTENCE_RE.split(text) if s and s.strip()]


def _pack(units: list[str], max_chars: int, separator: str) -> list[str]:
    """Gộp các đơn vị liền kề thành block ≤ max_chars.

    Không bao giờ cắt giữa một đơn vị — đơn vị dài hơn max_chars được giữ nguyên,
    vì cắt ngang câu còn hại hơn là chunk hơi quá cỡ.
    """
    blocks, current = [], ""
    for unit in units:
        if current and len(current) + len(separator) + len(unit) > max_chars:
            blocks.append(current)
            current = unit
        else:
            current = f"{current}{separator}{unit}" if current else unit
    if current:
        blocks.append(current)
    return blocks


def _get_encoder():
    """Lazy-load encoder rồi cache — load model tốn vài giây, mỗi process 1 lần là đủ."""
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer

        _encoder = SentenceTransformer(SEMANTIC_MODEL)
    return _encoder


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.

    Cắt chunk mới đúng chỗ cosine similarity giữa hai câu liền kề tụt xuống dưới
    `threshold` — đó là điểm văn bản đổi chủ đề.
    """
    metadata = metadata or {}
    sentences = _split_sentences(text)
    if not sentences:
        return []

    # normalize_embeddings=True → tích vô hướng chính là cosine similarity.
    embeddings = _get_encoder().encode(sentences, normalize_embeddings=True)
    similarities = (embeddings[:-1] * embeddings[1:]).sum(axis=1)

    groups, current = [], [sentences[0]]
    for sentence, similarity in zip(sentences[1:], similarities):
        topic_shift = similarity < threshold
        long_enough = len(" ".join(current)) >= SEMANTIC_MIN_CHARS
        if topic_shift and long_enough:
            groups.append(current)
            current = [sentence]
        else:
            current.append(sentence)
    groups.append(current)

    return [
        Chunk(text=" ".join(group),
              metadata={**metadata, "strategy": "semantic", "chunk_index": i})
        for i, group in enumerate(groups)
    ]


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Child mang sẵn `parent_text` trong metadata để pipeline mở rộng ngữ cảnh tại chỗ,
    không cần tra ngược một parent store riêng.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    parents: list[Chunk] = []
    children: list[Chunk] = []

    for parent_text in _pack(paragraphs, parent_size, "\n\n"):
        parent_id = f"parent_{len(parents)}"
        parents.append(Chunk(
            text=parent_text,
            metadata={**metadata, "chunk_type": "parent", "parent_id": parent_id},
        ))
        for child_text in _pack(_split_sentences(parent_text), child_size, " "):
            children.append(Chunk(
                text=child_text,
                metadata={**metadata, "chunk_type": "child", "parent_id": parent_id,
                          "parent_text": parent_text, "chunk_index": len(children)},
                parent_id=parent_id,
            ))

    return parents, children


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.

    Mỗi chunk mang breadcrumb đầy đủ (H1 > H2 > H3) ở đầu text: chunk "12 ngày làm việc"
    là vô nghĩa nếu tách khỏi heading "Nghỉ phép năm".
    """
    metadata = metadata or {}
    parts = re.split(r"(^#{1,3}\s+.+$)", text, flags=re.MULTILINE)

    chunks: list[Chunk] = []
    path: list[str] = []  # heading stack, index = level - 1

    for part in parts:
        header = _HEADER_RE.match(part or "")
        if header:
            del path[len(header.group(1)) - 1:]  # đóng các section con đang mở
            path.append(part.strip())
            continue

        body = (part or "").strip()
        if not body:
            continue  # heading rỗng nội dung → để dành làm breadcrumb cho chunk sau

        titles = [h.lstrip("# ").strip() for h in path]
        chunks.append(Chunk(
            text="\n\n".join(path + [body]),
            metadata={**metadata, "strategy": "structure", "chunk_index": len(chunks),
                      "section": titles[-1] if titles else "",
                      "section_path": " > ".join(titles)},
        ))

    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.
    (Đã implement sẵn — sẽ hoạt động khi bạn implement 3 strategies ở trên)
    """
    def _stats(chunk_list):
        lengths = [len(c.text) for c in chunk_list]
        if not lengths:
            return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        return {
            "count": len(lengths),
            "avg_len": round(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }

    all_text = "\n\n".join(d["text"] for d in documents)
    meta = {"source": "all"}

    basic = chunk_basic(all_text, metadata=meta)
    semantic = chunk_semantic(all_text, metadata=meta)
    parents, children = chunk_hierarchical(all_text, metadata=meta)
    structure = chunk_structure_aware(all_text, metadata=meta)

    results = {
        "basic": _stats(basic),
        "semantic": _stats(semantic),
        "hierarchical": {**_stats(children), "parents": len(parents)},
        "structure": _stats(structure),
    }

    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, s in results.items():
        print(f"{name:<15} {s['count']:>7} {s['avg_len']:>5} {s['min_len']:>5} {s['max_len']:>5}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
