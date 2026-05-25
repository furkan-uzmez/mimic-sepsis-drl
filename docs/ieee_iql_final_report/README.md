# IEEE IQL Final Report Build Notes

Generated from the IEEE report skill using the protocol, gap analysis, and current IQL final artifacts.

## Files

- `iql_final_ieee_report.tex`: IEEEtran LaTeX report.
- `references.bib`: BibTeX references used by the report.

## Build

Run from this directory:

```bash
pdflatex iql_final_ieee_report.tex
bibtex iql_final_ieee_report
pdflatex iql_final_ieee_report.tex
pdflatex iql_final_ieee_report.tex
```

The report uses figures from `../../results/iql_final/figures/`. If the report is moved, update the relative `\includegraphics` paths.

## Important Interpretation Note

The final artifact bundle currently contains non-estimable selected-policy FQE/WIS values. The report therefore presents the workflow as structurally complete but does not claim clinical or statistical superiority for the selected IQL policy.
