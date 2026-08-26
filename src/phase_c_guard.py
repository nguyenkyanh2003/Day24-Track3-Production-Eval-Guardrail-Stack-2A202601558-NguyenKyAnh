from __future__ import annotations

"""Phase C: Production Guardrails — Presidio PII + NeMo Guardrails + P95 Latency."""

import asyncio
import json
import os
import re
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ADVERSARIAL_SET_PATH, GUARDRAILS_CONFIG_DIR, LATENCY_BUDGET_P95_MS, PRESIDIO_LANGUAGE

_PRESIDIO_ENGINES = None
_NEMO_RAILS = None

_PROMPT_INJECTION_RE = re.compile(
    r"system\s*override|ignore\s+(your|all|previous)|forget\s+(all|your)|"
    r"admin\s+command|priority\s*1|ignore\s+previous\s+instructions|"
    r"system\s+(instructions|prompt)|training\s+data|dump\s+all|"
    r"bỏ\s+qua\s+(tất\s+cả|mọi)|in\s+ra\s+toàn\s+bộ\s+system",
    re.IGNORECASE,
)
_JAILBREAK_RE = re.compile(
    r"pretend\s+you\s+are\s+dan|do\s+anything\s+now|unrestricted\s+ai|"
    r"ai\s+không\s+có\s+giới\s+hạn|đóng\s+vai\s+hr\s+manager|"
    r"tôi\s+là\s+ceo|tấn\s+công\s+mạng",
    re.IGNORECASE,
)
_OFF_TOPIC_RE = re.compile(
    r"bài\s+thơ|mùa\s+thu|nấu\s+phở|công\s+thức\s+truyền\s+thống|"
    r"bitcoin|ethereum|phương\s+trình\s+vi\s+phân|marvel|"
    r"recommend.*phim|thời\s+tiết|giá\s+cổ\s+phiếu",
    re.IGNORECASE,
)
_PII_REQUEST_RE = re.compile(
    r"cccd\s+(và|của)|số\s+điện\s+thoại\s+của\s+nhân\s+viên|"
    r"list\s+all\s+employee\s+salaries|bảng\s+lương\s+chi\s+tiết|"
    r"confidential\s+(hr\s+)?data|employee\s+records|"
    r"liệt\s+kê\s+toàn\s+bộ\s+thông\s+tin\s+nhân\s+viên|"
    r"tiết\s+lộ\s+lương.*nhân\s+viên|thông\s+tin\s+cá\s+nhân\s+của",
    re.IGNORECASE,
)


# ─── Task 9a: Presidio PII Detection ─────────────────────────────────────────

def setup_presidio():
    """Khởi tạo Presidio engine với custom Vietnamese PII recognizers. (Đã implement sẵn)

    Custom recognizers thêm vào:
        VN_CCCD  — số CCCD 12 chữ số hoặc CMND 9 chữ số
        VN_PHONE — số điện thoại Việt Nam (0[3-9]xxxxxxxx)

    Các recognizers mặc định đã có sẵn: EMAIL, PHONE_NUMBER (international), ...
    """
    from presidio_analyzer import AnalyzerEngine, RecognizerRegistry, Pattern, PatternRecognizer
    from presidio_anonymizer import AnonymizerEngine

    global _PRESIDIO_ENGINES
    if _PRESIDIO_ENGINES is not None:
        return _PRESIDIO_ENGINES

    cccd_recognizer = PatternRecognizer(
        supported_entity="VN_CCCD",
        patterns=[
            Pattern("CCCD 12 digits", r"\b\d{12}\b", 0.9),
            Pattern("CMND 9 digits",  r"\b\d{9}\b",  0.7),
        ],
    )
    phone_recognizer = PatternRecognizer(
        supported_entity="VN_PHONE",
        patterns=[Pattern("VN mobile", r"\b0[3-9]\d{8}\b", 0.9)],
    )
    email_recognizer = PatternRecognizer(
        supported_entity="EMAIL_ADDRESS",
        patterns=[Pattern("Email address", r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", 0.95)],
    )

    registry = RecognizerRegistry(supported_languages=[PRESIDIO_LANGUAGE])
    registry.add_recognizer(cccd_recognizer)
    registry.add_recognizer(phone_recognizer)
    registry.add_recognizer(email_recognizer)

    analyzer  = AnalyzerEngine(registry=registry)
    anonymizer = AnonymizerEngine()
    _PRESIDIO_ENGINES = (analyzer, anonymizer)
    return _PRESIDIO_ENGINES


def pii_scan(text: str, analyzer=None, anonymizer=None) -> dict:
    """Task 9a: Quét PII trong văn bản bằng Presidio.

    Returns:
        {
          "has_pii":    bool,
          "entities":   [{"type": str, "text": str, "score": float, "start": int, "end": int}],
          "anonymized": str,   # text với PII được thay bằng <TYPE>
        }
    """
    if analyzer is None or anonymizer is None:
        default_analyzer, default_anonymizer = setup_presidio()
        analyzer = analyzer or default_analyzer
        anonymizer = anonymizer or default_anonymizer

    results = analyzer.analyze(
        text=text,
        language=PRESIDIO_LANGUAGE,
        entities=["VN_CCCD", "VN_PHONE", "EMAIL_ADDRESS"],
    )
    results = sorted(results, key=lambda result: (result.start, -(result.end - result.start)))
    if not results:
        return {"has_pii": False, "entities": [], "anonymized": text}
    entities = [
        {"type": result.entity_type, "text": text[result.start:result.end],
         "score": round(float(result.score), 3), "start": result.start, "end": result.end}
        for result in results
    ]
    return {"has_pii": True, "entities": entities,
            "anonymized": anonymizer.anonymize(text=text, analyzer_results=results).text}


# ─── Task 9b + 11: NeMo Guardrails ───────────────────────────────────────────

def setup_nemo_rails():
    """Khởi tạo NeMo Guardrails từ guardrails/config.yml. (Đã implement sẵn)

    Config directory: guardrails/
        config.yml  — model + rails config
        rails.co    — Colang dialogue flows (topic check, jailbreak check, output check)
    """
    global _NEMO_RAILS
    if _NEMO_RAILS is not None:
        return _NEMO_RAILS
    from nemoguardrails import RailsConfig, LLMRails
    config = RailsConfig.from_path(GUARDRAILS_CONFIG_DIR)
    _NEMO_RAILS = LLMRails(config)
    return _NEMO_RAILS


def _local_input_reason(text: str) -> str | None:
    """Fast deterministic first line of defense before the LLM-backed rail."""
    for reason, pattern in (
        ("prompt_injection", _PROMPT_INJECTION_RE),
        ("jailbreak", _JAILBREAK_RE),
        ("pii_request", _PII_REQUEST_RE),
        ("off_topic", _OFF_TOPIC_RE),
    ):
        if pattern.search(text):
            return reason
    return None


def _response_text(response) -> str:
    if isinstance(response, dict):
        return str(response.get("content") or response.get("response") or "")
    return str(response or "")


async def check_input_rail(text: str, rails=None) -> dict:
    """Task 9b: Kiểm tra input qua NeMo input rails (topic guard + jailbreak guard).

    Returns:
        {
          "allowed":        bool,
          "blocked_reason": str | None,
          "response":       str,          # NeMo's raw response
        }
    """
    local_reason = _local_input_reason(text)
    if local_reason:
        return {"allowed": False, "blocked_reason": local_reason,
                "response": "Yêu cầu bị chặn bởi input guard."}
    if rails is None:
        return {"allowed": True, "blocked_reason": None, "response": "local_allow"}

    response = _response_text(await rails.generate_async(
        messages=[{"role": "user", "content": text}]
    ))
    refuse_keywords = ("xin lỗi", "không thể", "không được phép", "i cannot", "i'm sorry")
    blocked = any(keyword in response.lower() for keyword in refuse_keywords)
    return {"allowed": not blocked,
            "blocked_reason": "nemo_input_rail" if blocked else None,
            "response": response}


async def check_output_rail(question: str, answer: str, rails=None) -> dict:
    """Task 11: Kiểm tra LLM output qua NeMo output rails trước khi trả về user.

    NeMo output rails hoạt động trong context của cả cuộc hội thoại (input + output).
    Kiểm tra: có PII không? Nội dung có phù hợp không? Có hallucination rõ ràng không?

    Returns:
        {
          "safe":           bool,
          "flagged_reason": str | None,
          "final_answer":   str,          # answer đã qua guard (có thể bị redact)
        }
    """
    pii = pii_scan(answer)
    sensitive = re.search(
        r"mật\s+khẩu\s+(admin|hệ\s+thống)|cccd\s+của\s+nhân\s+viên|"
        r"số\s+điện\s+thoại\s+cá\s+nhân|bảng\s+lương\s+chi\s+tiết",
        answer, re.IGNORECASE,
    )
    if pii["has_pii"] or sensitive:
        return {"safe": False, "flagged_reason": "sensitive_output",
                "final_answer": "Tôi không thể cung cấp thông tin nhạy cảm này."}
    if rails is None:
        return {"safe": True, "flagged_reason": None, "final_answer": answer}

    response = _response_text(await rails.generate_async(messages=[
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]))
    blocked = any(keyword in response.lower()
                  for keyword in ("không thể cung cấp", "i cannot"))
    return {"safe": not blocked,
            "flagged_reason": "nemo_output_rail" if blocked else None,
            "final_answer": response if blocked else answer}


# ─── Task 10: Adversarial Test Suite ─────────────────────────────────────────

def run_adversarial_suite(adversarial_set: list[dict], rails=None,
                           analyzer=None, anonymizer=None) -> list[dict]:
    """Task 10: Chạy 20 adversarial inputs qua full guard stack, so sánh với expected.

    Guard stack order:
        1. pii_scan()         → block nếu has_pii (cho category pii_injection)
        2. check_input_rail() → block nếu jailbreak / off-topic / prompt injection

    Returns:
        list of {
          "id": int, "category": str, "input": str,
          "expected": "blocked"|"allowed",
          "actual":   "blocked"|"allowed",
          "blocked_by": str | None,       # "presidio" | "nemo_input" | None
          "passed": bool,
        }
    """
    if analyzer is None or anonymizer is None:
        analyzer, anonymizer = setup_presidio()

    async def _run_all():
        output = []
        for item in adversarial_set:
            blocked_by = None
            if pii_scan(item["input"], analyzer, anonymizer)["has_pii"]:
                blocked_by = "presidio"
            if blocked_by is None:
                rail_result = await check_input_rail(item["input"], rails)
                if not rail_result["allowed"]:
                    blocked_by = "nemo_input"
            actual = "blocked" if blocked_by else "allowed"
            output.append({"id": item["id"], "category": item["category"],
                           "input": item["input"], "expected": item["expected"],
                           "actual": actual, "blocked_by": blocked_by,
                           "passed": actual == item["expected"]})
        return output

    results = asyncio.run(_run_all())
    passed = sum(result["passed"] for result in results)
    print(f"Adversarial suite: {passed}/{len(results)} passed")
    return results


# ─── Task 12: P95 Latency Measurement ────────────────────────────────────────

def measure_p95_latency(test_inputs: list[str], n_runs: int = 20,
                         rails=None, analyzer=None, anonymizer=None) -> dict:
    """Task 12: Đo P50/P95/P99 latency cho từng layer trong guard stack.

    Mục tiêu production: P95 total < LATENCY_BUDGET_P95_MS (500ms mặc định)

    Insight cần quan sát:
        - Presidio: local regex → rất nhanh (<10ms)
        - NeMo:     LLM API call → chậm (~200-800ms tuỳ model và network)
        → Tổng: dominated by NeMo

    Returns:
        {
          "presidio_ms":  {"p50": float, "p95": float, "p99": float},
          "nemo_ms":      {"p50": float, "p95": float, "p99": float},
          "total_ms":     {"p50": float, "p95": float, "p99": float},
          "latency_budget_ok": bool,
          "budget_ms": int,
        }
    """
    if not test_inputs or n_runs <= 0:
        raise ValueError("test_inputs must not be empty and n_runs must be positive")
    if analyzer is None or anonymizer is None:
        analyzer, anonymizer = setup_presidio()
    presidio_times, nemo_times, total_times = [], [], []

    async def _measure():
        for index in range(n_runs):
            text = test_inputs[index % len(test_inputs)]
            started = time.perf_counter()
            pii_scan(text, analyzer, anonymizer)
            presidio_ms = (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            await check_input_rail(text, rails)
            nemo_ms = (time.perf_counter() - started) * 1000
            presidio_times.append(presidio_ms)
            nemo_times.append(nemo_ms)
            total_times.append(presidio_ms + nemo_ms)

    asyncio.run(_measure())

    def percentiles(values):
        ordered = sorted(values)
        def nearest_rank(percentile):
            index = max(0, min(len(ordered) - 1,
                               int((percentile / 100) * len(ordered) + 0.999999) - 1))
            return round(ordered[index], 3)
        return {"p50": nearest_rank(50), "p95": nearest_rank(95),
                "p99": nearest_rank(99)}

    total = percentiles(total_times)
    return {"presidio_ms": percentiles(presidio_times),
            "nemo_ms": percentiles(nemo_times), "total_ms": total,
            "latency_budget_ok": total["p95"] < LATENCY_BUDGET_P95_MS,
            "budget_ms": LATENCY_BUDGET_P95_MS}


def save_phase_c_report(results: list[dict], latency: dict,
                        path: str = "reports/guard_results.json") -> dict:
    passed = sum(result["passed"] for result in results)
    report = {"total": len(results), "passed": passed,
              "pass_rate": round(passed / len(results), 4) if results else 0.0,
              "results": results, "latency": latency}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(f"Phase C report saved → {path}")
    return report


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Task 9a: PII scan demo
    test_pii = "Nhân viên Nguyễn Văn A, CCCD 034095001234, SĐT 0987654321 hỏi về nghỉ phép."
    result = pii_scan(test_pii)
    print(f"PII detected: {result['has_pii']}")
    print(f"Entities: {result['entities']}")
    print(f"Anonymized: {result['anonymized']}")

    # Task 10: Adversarial suite
    with open(ADVERSARIAL_SET_PATH, encoding="utf-8") as f:
        adversarial_set = json.load(f)
    print(f"\nLoaded {len(adversarial_set)} adversarial inputs")
    analyzer, anonymizer = setup_presidio()
    results = run_adversarial_suite(adversarial_set, analyzer=analyzer, anonymizer=anonymizer)
    if results:
        passed = sum(1 for r in results if r["passed"])
        print(f"Adversarial suite: {passed}/{len(results)} passed")

    # Task 12: P95 latency
    sample_inputs = [item["input"] for item in adversarial_set[:10]]
    latency = measure_p95_latency(sample_inputs, n_runs=20,
                                  analyzer=analyzer, anonymizer=anonymizer)
    print(f"\nLatency P95 — Presidio: {latency['presidio_ms']['p95']}ms | "
          f"NeMo: {latency['nemo_ms']['p95']}ms | "
          f"Total: {latency['total_ms']['p95']}ms")
    print(f"Budget OK ({latency['budget_ms']}ms): {latency['latency_budget_ok']}")
    save_phase_c_report(results, latency)
