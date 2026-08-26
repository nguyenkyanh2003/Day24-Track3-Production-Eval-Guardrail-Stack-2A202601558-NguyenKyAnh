# CI/CD Blueprint: RAG Evaluation + Guardrail Stack

**Sinh viên:** Nguyễn Kỳ Anh<br>
**Ngày:** 26/08/2026<br>
**Release target:** HR policy assistant

## 1. Guard stack architecture

```text
User input
   │
   ├─ Presidio PII scan
   │    block: VN_CCCD, VN_PHONE, EMAIL, CMND
   │    action: 400 + generic PII refusal; never log raw PII
   │
   ├─ NeMo/local input rail
   │    block: jailbreak, prompt injection, PII request, off-topic
   │    action: safe refusal + reason code
   │
   ├─ Day 18 RAG pipeline
   │    hierarchical chunks → hybrid BM25+dense → rerank top 8 → top 3
   │    → Gemini answer grounded in source-labelled HR contexts
   │
   ├─ NeMo/local output rail
   │    block/redact: PII, credentials, employee-specific confidential data
   │    action: replace unsafe output with safe response
   │
   └─ User response + structured telemetry (no prompt/PII payload)
```

Fail-closed applies to explicit PII and injection signals. Provider timeout returns a controlled unavailable response; it must not bypass a rail. API/HF tokens remain in secret storage or local `.env`, never in Git or application logs.

## 2. Latency budget

Task 12 measured 20 input-guard runs. The NeMo figure below is the deterministic local fast path used by the suite; a remote LLM rail must be benchmarked separately before production.

| Measured layer | P50 (ms) | P95 (ms) | P99 (ms) | Gate |
|---|---:|---:|---:|---|
| Presidio PII | 14.752 | 19.170 | 24.281 | observe; warm recognizers once |
| NeMo/local input rail | 0.031 | 0.091 | 0.347 | <300 ms |
| **Measured input guard total** | **14.779** | **19.193** | **24.300** | **<500 ms: PASS** |
| RAG pipeline | not measured in Task 12 | not measured | not measured | target <2,000 ms |
| Output rail | not measured separately | not measured | not measured | target <300 ms |

The measured P95 is **19.193 ms**, safely inside the lab’s 500 ms guard budget. Presidio dominates the local path; construct and cache its analyzer/anonymizer at process startup. Production rollout must measure end-to-end P95/P99, including networked NeMo/Gemini and output scanning, rather than presenting the input-only number as full request latency.

## 3. CI/CD quality gates

```yaml
pull_request:
  - install pinned dependencies on Python 3.11
  - run: python -m pytest tests/ -q
  - require: 40/40 tests pass and zero TODO markers
  - run deterministic adversarial suite
  - require: adversarial_pass_rate >= 0.75
  - require: measured_input_guard_p95_ms < 500

nightly_or_release_candidate:
  - restore an isolated Qdrant snapshot and generate all 50 answers
  - run all four RAGAS metrics on 50/50 questions
  - require: total_questions == 50
  - require: overall_avg_score >= 0.65
  - require: faithfulness_by_distribution >= 0.70
  - run pairwise judge twice with swapped positions
  - require: cohen_kappa >= 0.60
  - archive: answers and three JSON reports with commit SHA/model version

deployment:
  - canary 5% traffic
  - verify guard latency, PII blocks and refusal-rate drift
  - promote only when SLOs hold; otherwise automatic rollback
```

LLM evaluations run nightly/release-candidate rather than on every PR because they require provider credentials, quota and non-deterministic external services. PR gates remain deterministic. Secrets are injected by the CI secret manager and forked pull requests cannot access them.

## 4. Production monitoring

| Signal | Threshold | Action |
|---|---:|---|
| Daily sampled RAGAS faithfulness | <0.70 | stop promotion; review retrieval/prompt |
| Overall RAGAS average | <0.65 or drop >0.08 | compare corpus/model version and rollback |
| Adversarial pass rate | <90% warning; <75% critical | security review and add regression cases |
| Position bias | >30% | require swap consensus; human-review ties |
| Calibration κ | <0.60 | disable judge as blocking gate |
| Input guard P95 | >500 ms | profile Presidio, capacity check, rollback |
| End-to-end P95 | >2,500 ms | isolate retrieval/provider bottleneck |
| PII events | >10/hour or sudden 3× spike | security alert; retain only entity/type counts |
| Provider errors/429 | >2% for 5 minutes | backoff, circuit breaker, fallback response |

Dashboard dimensions include release SHA, model, prompt version, corpus version, distribution, rail and reason code. Sampled human review covers allowed, blocked and judge-tie cases. Logs exclude raw questions, answers, access tokens and detected identifiers; access is least-privilege with a defined retention window.

## 5. Lab evidence and release decision

| Check | Actual result | Decision |
|---|---:|---|
| 50-question RAGAS average | 0.8411 | PASS |
| Factual / multi-hop / adversarial | 0.9444 / 0.7292 / 0.8582 | PASS; improve multi-hop |
| Dominant failure | multi-hop / answer_relevancy | track remediation |
| Cohen’s κ / agreement | 1.0000 / 100% | PASS |
| Position bias | 10% | PASS; retain swap |
| Adversarial guard suite | 20/20 (100%) | PASS + bonus |
| Measured input-guard P95 | 19.193 ms | PASS |
| Automated tests | 40/40 | PASS |

Release recommendation: **pass for lab submission and controlled canary**, not immediate unrestricted production. The strongest next improvement is query decomposition plus version metadata filtering for multi-hop retrieval. Before a real rollout, enlarge the human calibration set, benchmark remote rail and RAG latency, add load/failure-injection tests, and establish an incident runbook.
