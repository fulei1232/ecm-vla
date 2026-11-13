#!/usr/bin/env bash
# Build build/main.pdf and build/supplementary.pdf with the portable TinyTeX in /media/fulei/fuleivla/tools/.TinyTeX
# (elsarticle added to its texmf-local). Overleaf: upload this folder plus ../figures and set main.tex as root.
set -euo pipefail
cd "$(dirname "$0")"
export PATH=/media/fulei/fuleivla/tools/.TinyTeX/bin/x86_64-linux:$PATH
mkdir -p build
cp references.bib build/
for doc in main supplementary; do
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build $doc.tex > build/$doc.pass1.log
  if grep -q "\\\\bibliography{" $doc.tex; then (cd build && bibtex $doc > $doc.bibtex.log); fi
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build $doc.tex > build/$doc.pass2.log
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build $doc.tex > build/$doc.pass3.log
  grep -E "undefined|multiply defined" build/$doc.log || true
  echo "build/$doc.pdf"
done
