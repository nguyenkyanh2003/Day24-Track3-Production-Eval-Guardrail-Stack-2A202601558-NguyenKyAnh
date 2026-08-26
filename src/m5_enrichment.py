from __future__ import annotations

"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import json
import os
import re
import sys
import threading
import time
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LLM_API_KEY, LLM_MAX_RPM, LLM_MODEL, OPENAI_BASE_URL

ENRICH_MODEL = LLM_MODEL
DEFAULT_METADATA = {"topic": "general", "entities": [], "category": "policy", "language": "vi"}

_client = None
_last_call_at = 0.0
_call_lock = threading.Lock()


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


# ─── LLM helpers ─────────────────────────────────────────


def _get_client():
    """Lazy singleton — tránh dựng lại HTTP client cho từng chunk."""
    global _client
    if _client is None:
        from openai import OpenAI

        _client = OpenAI(api_key=LLM_API_KEY, base_url=OPENAI_BASE_URL or None, max_retries=5)
    return _client


def wait_for_rate_limit() -> None:
    """Giãn các call cho cách nhau đủ 60/LLM_MAX_RPM giây.

    Enrichment gọi API trong vòng lặp sát nhau; free tier Gemini chặn 15 request/phút
    nên nếu không giãn thì phần lớn chunk rơi hết về fallback extractive.
    """
    global _last_call_at
    if LLM_MAX_RPM <= 0:
        return

    min_interval = 60.0 / LLM_MAX_RPM
    with _call_lock:
        idle = time.monotonic() - _last_call_at
        if idle < min_interval:
            time.sleep(min_interval - idle)
        _last_call_at = time.monotonic()


def _chat(system: str, user: str, max_tokens: int, json_mode: bool = False) -> str | None:
    """Gọi 1 lượt chat completion.

    Trả về None khi không có API key hoặc call lỗi — mọi technique bên dưới đều có
    fallback extractive, nên lab vẫn chạy được offline.
    """
    if not LLM_API_KEY:
        return None

    wait_for_rate_limit()
    try:
        response = _get_client().chat.completions.create(
            model=ENRICH_MODEL,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=0,
            **({"response_format": {"type": "json_object"}} if json_mode else {}),
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as e:  # noqa: BLE001 - 1 chunk lỗi không được làm hỏng cả batch
        print(f"  ⚠️  OpenAI enrichment call failed: {e}")
        return None


def _parse_json(raw: str | None) -> dict:
    """Parse JSON từ LLM. Trả về {} nếu không parse được."""
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _extractive_summary(text: str) -> str:
    """Fallback không cần API: lấy 2 câu đầu."""
    sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
    return ". ".join(sentences[:2]).rstrip(".") + "." if sentences else text


def _extractive_questions(text: str, n_questions: int) -> list[str]:
    """Fallback không cần API: biến các câu khẳng định thành câu hỏi thô."""
    sentences = [s.strip() for s in re.split(r"[.!?\n]", text) if len(s.strip()) > 10]
    return [f"{s.rstrip('.')}?" for s in sentences[:n_questions]]


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.
    """
    summary = _chat(
        "Tóm tắt đoạn văn sau trong 2-3 câu ngắn gọn bằng tiếng Việt.",
        text,
        max_tokens=150,
    )
    # Bản tóm tắt dài hơn bản gốc thì không còn là tóm tắt → quay về extractive.
    if summary and len(summary) < len(text):
        return summary
    return _extractive_summary(text)


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).
    """
    raw = _chat(
        f"Dựa trên đoạn văn, tạo {n_questions} câu hỏi mà đoạn văn có thể trả lời. "
        "Trả về mỗi câu hỏi trên 1 dòng, không đánh số.",
        text,
        max_tokens=200,
    )
    if raw:
        questions = [line.strip().lstrip("0123456789.-) ") for line in raw.splitlines() if line.strip()]
        if questions:
            return questions[:n_questions]
    return _extractive_questions(text, n_questions)


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).
    """
    context = _chat(
        "Viết 1 câu ngắn mô tả đoạn văn này nằm ở đâu trong tài liệu và nói về chủ đề gì. "
        "Chỉ trả về 1 câu.",
        f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}",
        max_tokens=80,
    )
    if not context:
        context = f"Trích từ {document_title}." if document_title else ""
    return f"{context}\n\n{text}" if context else text


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.
    """
    parsed = _parse_json(_chat(
        'Trích xuất metadata từ đoạn văn. Trả về JSON: '
        '{"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}',
        text,
        max_tokens=150,
        json_mode=True,
    ))
    return {**DEFAULT_METADATA, **parsed}


# ─── Combined Single-Call Mode ───────────────────────────


_COMBINED_PROMPT = """Phân tích đoạn văn và trả về JSON:
{
  "summary": "tóm tắt 2-3 câu",
  "questions": ["câu hỏi 1", "câu hỏi 2", "câu hỏi 3"],
  "context": "1 câu mô tả đoạn văn nằm ở đâu trong tài liệu",
  "metadata": {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}
}"""


def _enrich_single_call(text: str, source: str) -> dict:
    """Single LLM call to get summary + questions + context + metadata.

    ⚠️ Cost optimization: 1 API call thay vì 4 calls riêng lẻ.

    Không có API key (hoặc call lỗi) → ghép kết quả từ các fallback extractive để
    pipeline vẫn ra được enriched_text khác original_text.
    """
    parsed = _parse_json(_chat(
        _COMBINED_PROMPT,
        f"Tài liệu: {source}\n\nĐoạn văn:\n{text}",
        max_tokens=400,
        json_mode=True,
    ))
    if parsed:
        return parsed

    return {
        "summary": _extractive_summary(text),
        "questions": _extractive_questions(text, 3),
        "context": f"Trích từ {source}." if source else "",
        "metadata": dict(DEFAULT_METADATA),
    }


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks. (Đã implement sẵn — dùng functions ở trên)

    Có 2 chế độ:
    - methods cụ thể (["summary"], ["contextual"]...): gọi từng function riêng (tốt cho học/debug)
    - methods=["combined"] hoặc None: 1 API call duy nhất cho tất cả (tốt cho production)

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: Default None → combined mode (1 call/chunk).
                 Options: "summary", "hyqa", "contextual", "metadata", "combined"
    """
    if methods is None:
        methods = ["combined"]

    use_combined = "combined" in methods

    enriched = []
    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        source = chunk.get("metadata", {}).get("source", "")

        if use_combined:
            result = _enrich_single_call(text, source)
            summary = result.get("summary", "")
            questions = result.get("questions", [])
            context_line = result.get("context", "")
            enriched_text = f"{context_line}\n\n{text}" if context_line else text
            auto_meta = result.get("metadata", {})
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            enriched_text = contextual_prepend(text, source) if "contextual" in methods else text
            auto_meta = extract_metadata(text) if "metadata" in methods else {}

        enriched.append(EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**chunk.get("metadata", {}), **auto_meta},
            method="+".join(methods),
        ))

        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            print(f"  Enriched {i + 1}/{len(chunks)} chunks...", flush=True)

    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
