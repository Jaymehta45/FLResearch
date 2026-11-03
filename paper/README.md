# IEEE Sample Research Paper

This folder contains an IEEE conference-style sample paper generated from the FLResearch project experiments.

## Files

| File | Description |
|------|-------------|
| `ieee_paper.pdf` | Ready-to-read PDF (recommended) |
| `ieee_paper.html` | Printable IEEE-style HTML (Print → Save as PDF) |
| `ieee_paper.tex` | LaTeX source (`IEEEtran` conference class) |
| `figures/` | All paper figures (architecture + experiment + confusion) |

## Contents covered

- Abstract, keywords, introduction, related work
- Methodology: dataset, MLP, FedAvg, label poisoning
- Architecture diagrams (pipeline, FedAvg, MLP, poison attack)
- 64-run experiment results with charts and Table I
- Confusion matrices + TP/FP/FN/TN analysis
- Discussion, conclusion, IEEE-style references
- Link to live dashboard: https://fl-research.vercel.app

## Compile LaTeX (optional)

If you have TeX installed:

```bash
cd paper
pdflatex ieee_paper.tex
pdflatex ieee_paper.tex
```

Requires the `IEEEtran` class (usually included with TeX Live / MacTeX).

## Note

This is a **sample / draft research paper** grounded in the project’s real experimental outputs. Update author contact details before any formal submission.
