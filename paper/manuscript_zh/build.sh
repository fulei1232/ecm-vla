#!/usr/bin/env bash
# Regenerates values.tex, the figures and main.pdf.  Usage: ./build.sh [--no-figures]
set -euo pipefail
cd "$(dirname "$0")"
export PATH=/media/fulei/fuleivla/tools/.TinyTeX/bin/x86_64-linux:$PATH
PY=/media/fulei/fuleivla/robomme_benchmark/.venv/bin/python
$PY tools/gen_values.py
if [[ "${1:-}" != "--no-figures" ]]; then $PY tools/make_figures_zh.py; fi
mkdir -p build
xelatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex > build/pass1.log
cp references.bib build/
(cd build && bibtex main > bibtex.log)
xelatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex > build/pass2.log
xelatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex > build/pass3.log
grep -E "Warning: (Reference|Citation).*undefined|Undefined value" build/main.log || true
cp build/main.pdf HarvestMem_zh_draft.pdf
echo "built $(pwd)/HarvestMem_zh_draft.pdf"
