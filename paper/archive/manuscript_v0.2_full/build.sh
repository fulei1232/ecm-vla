#!/usr/bin/env bash
# Build build/main.pdf with the portable TinyTeX in /media/fulei/fuleivla/tools/.TinyTeX
# (elsarticle added to its texmf-local). Overleaf: upload this folder plus ../figures and set main.tex as root.
set -euo pipefail
cd "$(dirname "$0")"
export PATH=/media/fulei/fuleivla/tools/.TinyTeX/bin/x86_64-linux:$PATH
mkdir -p build
cp references.bib build/
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex > build/pass1.log
(cd build && bibtex main > bibtex.log)
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex > build/pass2.log
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex > build/pass3.log
grep -E "undefined|multiply defined" build/main.log || true
echo "build/main.pdf"
