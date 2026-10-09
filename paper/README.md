# Paper

`main.tex` is IEEEtran two-column, 6–8 pages, targeting 15–25 references mostly
from the last five years.

## The rule

**No number is typed into this document by hand.** Tables are `\input` from
`results/`, which `experiments/aggregate.py` generates. `claims.csv` maps each
quantitative sentence to the results cell behind it, and the paper agent fails
if a claim has no backing number or disagrees with one.

This is not bureaucracy: it is the only way the reproducibility claim in the
paper is true, and it makes a stale number after a re-run impossible to miss.

## Build

```bash
python -m experiments.aggregate      # regenerate results/*.tex first
cd paper && latexmk -pdf main.tex
```

`IEEEtran.cls` is not committed — download it from the IEEE template page, or
use Overleaf's IEEE Conference template and paste `main.tex` in.

## Figures

`figs/` is generated, not drawn. Anything hand-made belongs in `figs/manual/`
with a note saying why it could not be generated.

## claims.csv

A real CSV with a real header row — not a commented one. `csv.DictReader`
treats the first non-comment line as the header, so a `#`-prefixed header makes
it silently consume the first data row instead. Keep notes in this README.
