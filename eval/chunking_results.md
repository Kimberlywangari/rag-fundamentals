# Chunking experiment results

22 answerable questions; embedding model `all-MiniLM-L6-v2` (256-token input limit); cosine similarity; exact search.
A hit = a top-k chunk from the labelled PDF that contains the verbatim anchor phrase.
With so few questions one question is worth 4.5 percentage points, so differences of one question are noise.

Coverage = share of questions whose anchor phrase is intact inside at least one chunk (an upper bound on any hit@k).

| config | chunks | median chars | median tokens | % > 256 tok | coverage | hit@1 | hit@3 | hit@5 | hit@10 | MRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed 300/0 | 764 | 297 | 61 | 0.0 | 0.73 | 0.18 | 0.32 | 0.46 | 0.55 | 0.282 |
| fixed 300/30 | 843 | 297 | 62 | 0.0 | 0.82 | 0.23 | 0.41 | 0.46 | 0.50 | 0.328 |
| fixed 300/60 | 947 | 297 | 62 | 0.0 | 1.00 | 0.27 | 0.41 | 0.46 | 0.68 | 0.384 |
| fixed 300/90 | 1082 | 297 | 61 | 0.0 | 0.95 | 0.32 | 0.46 | 0.50 | 0.59 | 0.396 |
| fixed 600/0 | 382 | 597 | 121 | 0.0 | 0.73 | 0.36 | 0.41 | 0.50 | 0.59 | 0.418 |
| fixed 600/60 | 422 | 597 | 121 | 0.0 | 0.95 | 0.41 | 0.64 | 0.68 | 0.77 | 0.514 |
| fixed 600/120 | 474 | 597 | 122 | 0.0 | 1.00 | 0.36 | 0.64 | 0.77 | 0.82 | 0.507 |
| fixed 600/180 | 542 | 597 | 122 | 0.0 | 1.00 | 0.36 | 0.64 | 0.64 | 0.82 | 0.496 |
| fixed 800/0 | 287 | 797 | 161 | 0.0 | 0.95 | 0.36 | 0.59 | 0.68 | 0.73 | 0.499 |
| fixed 800/80 | 317 | 797 | 161 | 0.0 | 0.95 | 0.46 | 0.59 | 0.68 | 0.82 | 0.556 |
| fixed 800/160 | 355 | 797 | 162 | 0.0 | 1.00 | 0.41 | 0.59 | 0.73 | 0.82 | 0.530 |
| fixed 800/240 | 405 | 798 | 161 | 0.0 | 1.00 | 0.59 | 0.68 | 0.73 | 0.86 | 0.646 |
| fixed 1000/0 | 229 | 997 | 201 | 0.0 | 0.95 | 0.36 | 0.59 | 0.73 | 0.82 | 0.502 |
| fixed 1000/100 | 254 | 997 | 201 | 0.0 | 1.00 | 0.41 | 0.73 | 0.73 | 0.86 | 0.588 |
| fixed 1000/200 | 284 | 998 | 202 | 0.0 | 1.00 | 0.59 | 0.82 | 0.91 | 0.95 | 0.724 |
| fixed 1000/300 | 325 | 997 | 201 | 0.0 | 1.00 | 0.46 | 0.73 | 0.86 | 0.86 | 0.605 |
| fixed 1200/0 | 191 | 1198 | 242 | 15.7 | 0.91 | 0.36 | 0.59 | 0.64 | 0.77 | 0.505 |
| fixed 1200/120 | 211 | 1197 | 242 | 18.5 | 1.00 | 0.50 | 0.73 | 0.95 | 0.95 | 0.658 |
| fixed 1200/240 | 238 | 1197 | 241 | 15.1 | 1.00 | 0.41 | 0.64 | 0.73 | 0.95 | 0.575 |
| fixed 1200/360 | 271 | 1197 | 242 | 18.1 | 1.00 | 0.59 | 0.77 | 0.82 | 0.86 | 0.686 |
| fixed 1500/0 | 153 | 1497 | 301 | 98.0 | 0.95 | 0.50 | 0.64 | 0.64 | 0.82 | 0.594 |
| fixed 1500/150 | 170 | 1497 | 301 | 98.2 | 1.00 | 0.46 | 0.64 | 0.73 | 0.86 | 0.576 |
| fixed 1500/300 | 190 | 1497 | 301 | 98.9 | 1.00 | 0.55 | 0.77 | 0.86 | 0.91 | 0.672 |
| fixed 1500/450 | 217 | 1497 | 301 | 99.1 | 1.00 | 0.59 | 0.77 | 0.77 | 0.91 | 0.687 |
| structure max 800 | 416 | 619 | 127 | 0.0 | 1.00 | 0.50 | 0.77 | 0.86 | 0.91 | 0.646 |
| structure max 1200 | 302 | 813 | 162 | 3.3 | 1.00 | 0.50 | 0.77 | 0.86 | 0.91 | 0.653 |
| structure max 1500 | 276 | 827 | 164 | 23.2 | 1.00 | 0.50 | 0.77 | 0.86 | 0.91 | 0.646 |
| structure max 2000 | 237 | 890 | 180 | 30.4 | 1.00 | 0.50 | 0.73 | 0.86 | 0.91 | 0.635 |

## Ranked by MRR (then hit@1, hit@3)

1. **fixed 1000/200**: MRR 0.724, hit@1 0.59, hit@3 0.82, hit@5 0.91
2. **fixed 1500/450**: MRR 0.687, hit@1 0.59, hit@3 0.77, hit@5 0.77
3. **fixed 1200/360**: MRR 0.686, hit@1 0.59, hit@3 0.77, hit@5 0.82
4. **fixed 1500/300**: MRR 0.672, hit@1 0.55, hit@3 0.77, hit@5 0.86
5. **fixed 1200/120**: MRR 0.658, hit@1 0.50, hit@3 0.73, hit@5 0.95

## Is **fixed 1000/200** really better? (paired bootstrap over questions)

| compared with | mean MRR difference | 95% interval | verdict |
|---|---:|---:|---|
| fixed 1000/0 | +0.222 | [+0.073, +0.376] | clearly better |
| structure max 1200 | +0.071 | [-0.054, +0.198] | not distinguishable |
| fixed 1500/450 | +0.037 | [-0.067, +0.143] | not distinguishable |

## Questions missed from the top 5 (config count out of 28)

- `dpa-07`: 28 configs
- `reg-03`: 22 configs
- `reg-05`: 22 configs
- `cmca-02`: 20 configs
- `cmca-05`: 15 configs
- `dpa-04`: 11 configs
- `cmca-03`: 10 configs
- `dpa-08`: 7 configs
- `cmca-04`: 7 configs
- `reg-06`: 6 configs
- `cmca-06`: 6 configs
- `dpa-03`: 4 configs
- `reg-02`: 3 configs
- `dpa-06`: 3 configs
- `dpa-01`: 3 configs
- `dpa-05`: 2 configs
- `cmca-01`: 2 configs
- `reg-07`: 1 configs
- `dpa-02`: 1 configs
- `reg-01`: 1 configs
