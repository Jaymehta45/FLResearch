# IEEE Sample Research Paper (Expanded)

This folder contains an expanded IEEE journal-style sample paper (~15–20 pages) generated from the FLResearch project experiments, with substantial theory.

## Files

| File | Description |
|------|-------------|
| `ieee_paper.pdf` | Ready-to-read PDF (~20 pages) |
| `ieee_paper.html` | Printable IEEE-style HTML |
| `ieee_paper.tex` | LaTeX source (`IEEEtran` **journal** class) |
| `figures/` | Architecture + experiment + confusion figures |

## Contents covered

- Extended abstract and keywords
- Introduction with problem formalization and contributions
- Background: IDS ML, FL, poisoning, UNSW-NB15
- **Theory:** risk minimization, client risk decomposition, FedAvg, communication complexity, MLP hypothesis class, imbalance metrics, threat model, dilution analysis, IID convergence intuition
- System design + Algorithm 1
- Experimental grid (64 runs) with descriptive statistics
- Results tables, charts, confusion matrices, TP/FP/FN/TN
- Discussion, limitations, future work, 10 references
- Dashboard: https://fl-research.vercel.app

## Compile LaTeX (optional)

```bash
cd paper
pdflatex ieee_paper.tex
pdflatex ieee_paper.tex
```

Requires `IEEEtran` (TeX Live / MacTeX).

## Note

Sample/draft paper grounded in real experimental outputs. Update author contact before formal submission.
