# NeuralVault Evaluation Report (my-project)

| Strategy | Recall@5 | MRR | nDCG@5 | Latency p50 (ms) | Latency p95 (ms) |
| --- | --- | --- | --- | --- | --- |
| bm25 | 0.8000 | 0.5417 | 1.1050 | 14.78 | 23.28 |
| vector | 0.8000 | 0.5417 | 1.1050 | 15.39 | 17.18 |
| hybrid | 0.8000 | 0.5417 | 1.1050 | 15.09 | 17.22 |
| hybrid+rerank | 0.8000 | 0.5167 | 1.0588 | 87.55 | 118.43 |
| full | 0.8000 | 0.5167 | 1.0588 | 88.78 | 118.42 |
