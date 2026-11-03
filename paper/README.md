# IEEE Sample Research Paper (Expanded)

Authentic **IEEE two-column journal format** (`IEEEtran`) compiled with TeX, plus theory, experiments, figures, and appendices.

## Files

| File | Description |
|------|-------------|
| `ieee_paper.pdf` | **Use this** — real IEEE two-column PDF |
| `ieee_paper.tex` | LaTeX source (`\documentclass[journal]{IEEEtran}`) |
| `ieee_paper.html` | Browser preview (not strict IEEE layout) |
| `build_pdf.py` | Alternate single-column ReportLab builder (not IEEE layout) |
| `figures/` | Architecture + experiment + confusion figures |

## Compile (IEEE format)

```bash
cd paper
tectonic ieee_paper.tex
# or: pdflatex ieee_paper.tex && pdflatex ieee_paper.tex
```

**Note:** IEEE two-column layout is dense. Roughly 9–12 IEEE pages ≈ 15–20 pages of single-column draft text.

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
