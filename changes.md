# Change Log

## [2026-11-03] - Fix blank IEEE PDF by rebuilding with ReportLab

### Summary
Replaced the blank Chrome headless PDF with a ReportLab-generated PDF that contains real selectable text, figures, tables, and page numbers.

### Reason
The previous `ieee_paper.pdf` from Chrome print-to-PDF had ~20 blank pages (no extractable text), so the paper appeared empty when opened.

### Files Modified
* `paper/build_pdf.py` — new PDF builder
* `paper/ieee_paper.pdf` — regenerated readable PDF (~16 pages, ~1.0 MB)
* `paper/README.md` — document rebuild command
* `changes.md` — this entry

### Implementation Details
Built the full expanded paper with ReportLab Platypus (Times fonts, justified body, figures from `paper/figures/`, Tables I–II, Algorithm 1, references). Verified with `pdftotext` that abstract/sections extract correctly.

### Impact
Users can now open and read the paper normally in Preview/Chrome/Acrobat.

### Testing
* `python3 build_pdf.py` succeeded
* `pdftotext` showed abstract, introduction, and contributions
* `pdfinfo`: 16 pages, ReportLab producer

### Additional Notes
Prefer Preview/Finder over opening the PDF as a text tab in Cursor.

---

## [2026-11-03] - Silence unusual line-terminator warning on PDF

### Summary
Added workspace editor settings so opening `ieee_paper.pdf` in Cursor/VS Code no longer prompts about unusual line terminators.

### Reason
PDFs are binary; the editor can mis-detect byte sequences (e.g. NEL `U+0085`) inside compressed streams as “unusual line terminators” when the file is opened as text.

### Files Modified
* `.vscode/settings.json` — set `editor.unusualLineTerminators` to `off`; exclude `*.pdf` from file watching noise

### Implementation Details
Confirmed `ieee_paper.html` has no U+2028/U+2029. The PDF has no LS/PS UTF-8 sequences; NEL bytes appear only as binary stream content and must not be stripped (that would corrupt the PDF).

### Impact
Editor UX only; paper content unchanged.

### Testing
* Inspected HTML/PDF byte contents for LS/PS/NEL.
* Did not alter PDF bytes.

### Additional Notes
Open `paper/ieee_paper.pdf` with Preview, Chrome, or Finder—not as a text tab—for normal reading.

---

## [2026-11-03] - Expanded IEEE paper to ~15–20 pages with theory

### Summary
Expanded the IEEE sample paper from a short conference draft into a ~20-page journal-style document with formal theory, algorithms, additional tables, deeper discussion, and more references.

### Reason
User requested a longer paper (15–20 pages) with more theory.

### Files Modified
* `paper/ieee_paper.tex` — switched to `IEEEtran` journal class; added theory sections (risk, FedAvg math, communication model, threat model, dilution, convergence), Algorithm 1, Table II, expanded related work/discussion, 10 references
* `paper/ieee_paper.html` — matching long-form printable HTML
* `paper/ieee_paper.pdf` — regenerated (~20 pages, ~1.4 MB)
* `paper/README.md` — updated documentation
* `changes.md` — this entry

### Implementation Details
* Added equations for population/empirical risk, client risk mixture, FedAvg weights, communication scaling, MLP map, accuracy/confusion/TP–FP–FN–TN, and poisoned-client dilution.
* Added mean-accuracy Table II from the 64-run sweep aggregates.
* Kept all architecture and experimental figures embedded in the PDF via HTML print pipeline.

### Impact
Paper length and theoretical depth increased substantially; experimental claims unchanged.

### Testing
* Regenerated PDF via headless Chrome; approximate page count ~20.
* Verified PDF/HTML/TEX file sizes on disk.

### Additional Notes
Still a sample/draft; author email placeholder remains.

---

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
