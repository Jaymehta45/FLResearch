# Change Log

## [2026-11-03] - IEEE sample research paper package

### Summary
Created a full IEEE conference-style sample research paper documenting the federated learning experiments on CIC–UNSW-NB15, including architecture diagrams, experiment charts, confusion matrices, results tables, and both PDF and LaTeX deliverables.

### Reason
User requested an IEEE-format sample paper covering the entire project with diagrams and results.

### Files Modified / Created
* `paper/ieee_paper.tex` — IEEEtran LaTeX source
* `paper/ieee_paper.html` — self-contained printable IEEE-style HTML
* `paper/ieee_paper.pdf` — generated PDF
* `paper/README.md` — paper folder instructions
* `paper/figures/fig_pipeline.png` — data pipeline diagram
* `paper/figures/fig_fedavg.png` — FedAvg round diagram
* `paper/figures/fig_mlp.png` — MLP architecture diagram
* `paper/figures/fig_poison.png` — label poisoning diagram
* `paper/figures/*.png` — copied experiment and confusion figures
* `changes.md` — this changelog

### Implementation Details
* Paper sections: Abstract, Introduction, Related Work, Methodology, Experimental Setup, Results, Discussion, Conclusion, References.
* Embedded project findings from 64-run sweep and confusion-cache scenarios.
* Generated architecture diagrams with Matplotlib.
* Produced PDF via headless Chrome print-to-PDF from the HTML version (local TeX not installed).

### Impact
Adds a submission-ready sample write-up for academic presentation; does not change training or dashboard code.

### Testing
* Verified `paper/ieee_paper.pdf` was written (~1.1 MB).
* Confirmed figure assets exist under `paper/figures/`.

### Additional Notes
Update author email / affiliation before any formal conference submission. LaTeX compile requires `IEEEtran` if regenerating PDF from `.tex`.
