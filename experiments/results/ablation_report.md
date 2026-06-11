# P0b Ablation Report

Questions: 36. Only completed variants are used for trade-off statements.

| Method | Correctness | Context Relevance | Citation Accuracy | Fallback Accuracy | Unsupported Claims | Supported Claim Ratio | Avg Retry | Avg Latency | Errors | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| V0 Naive RAG | 0.6111 | 1.0000 | 0.9615 | 0.8611 | N/A | N/A | 0.0000 | 4.4576 | 0 | completed |
| V1 + Query Transformation | 0.5833 | 1.0000 | 1.0000 | 0.8611 | N/A | N/A | 0.0000 | 8.5843 | 0 | completed |
| V2 + Retrieval Grading | 0.5833 | 1.0000 | 0.7308 | 0.7778 | N/A | N/A | 0.0000 | 13.8411 | 0 | completed |
| V3 + Conditional Retry / Fallback | 0.5833 | 0.9615 | 0.8077 | 0.7778 | N/A | N/A | 0.7778 | 25.6108 | 0 | completed |
| V4 + Hybrid Retrieval | 0.5833 | 0.9615 | 0.8462 | 0.8056 | N/A | N/A | 0.6944 | 25.7927 | 0 | completed |
| V5 + Reranker | 0.6389 | 0.9615 | 0.8077 | 0.8056 | N/A | N/A | 0.6667 | 28.6960 | 0 | completed |
| V6 + Claim-level Citation Verification | 0.5833 | 0.9615 | 0.7692 | 0.7778 | 0 | 1.0000 | 0.7500 | 41.2485 | 0 | completed |

## Observed Trade-offs

- V1 + Query Transformation vs V0 Naive RAG: correctness -0.0278; citation accuracy +0.0385; average latency +4.1267s.
- V2 + Retrieval Grading vs V1 + Query Transformation: citation accuracy -0.2692; fallback accuracy -0.0833; average latency +5.2568s.
- V3 + Conditional Retry / Fallback vs V2 + Retrieval Grading: context relevance -0.0385; citation accuracy +0.0769; average retry count +0.7778; average latency +11.7697s.
- V4 + Hybrid Retrieval vs V3 + Conditional Retry / Fallback: citation accuracy +0.0385; fallback accuracy +0.0278; average retry count -0.0834; average latency +0.1819s.
- V5 + Reranker vs V4 + Hybrid Retrieval: correctness +0.0556; citation accuracy -0.0385; average retry count -0.0277; average latency +2.9033s.
- V6 + Claim-level Citation Verification vs V5 + Reranker: correctness -0.0556; citation accuracy -0.0385; fallback accuracy -0.0278; average retry count +0.0833; average latency +12.5525s.

## Limitations

- Correctness is a deterministic keyword and gold-answer overlap heuristic.
- A single run does not provide confidence intervals or statistical significance.
- Citation verification metrics are N/A when that capability is disabled.
- Token usage and cost remain N/A unless the model client exposes reliable metadata.