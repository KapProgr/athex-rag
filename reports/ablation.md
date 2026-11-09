# Retrieval improvements (dev/test split)

Dev = questions on the 2024 reports (34 answerable), used to find failures and tune. Test = questions on the 2025 reports (43 answerable), only measured. BM25 with Greek stemming, k = 5. *answer@5*: a retrieved chunk states the answer; *hit@5*: a retrieved chunk is on a gold page.

| step | dev answer@5 | dev hit@5 | dev MRR | test answer@5 | test hit@5 | test MRR |
|---|---|---|---|---|---|---|
| baseline: BM25 + stemming, chunk 600 | 88.2% | 82.4% | 0.613 | 74.4% | 62.8% | 0.461 |
| + column-aware text | 88.2% | 82.4% | 0.635 | 74.4% | 62.8% | 0.460 |
| (+ page headings: not kept) | 88.2% | 82.4% | 0.615 | 74.4% | 60.5% | 0.433 |
| + glossary expansion | 94.1% | 88.2% | 0.659 | 83.7% | 74.4% | 0.510 |
| + BM25 k1=0.9, b=0.4 | 94.1% | 91.2% | 0.676 | 83.7% | 79.1% | 0.540 |

Each dev question is worth 2.9 points, each test question 2.3; differences of one or two questions are noise.
