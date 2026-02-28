.PHONY: all clean

all: main.pdf

main.pdf: main.tex experiments/folded_normal_py.pdf experiments/folded_normal_r.pdf experiments/gmm_py.pdf experiments/gmm_r.pdf
	pdflatex main.tex
	pdflatex main.tex

experiments/folded_normal_py.pdf experiments/gmm_py.pdf: experiments.py
	mkdir -p experiments
	python3 experiments.py

experiments/folded_normal_r.pdf experiments/gmm_r.pdf: experiments.R
	mkdir -p experiments
	Rscript experiments.R

clean:
	rm -f main.aux main.log main.out main.pdf
	rm -rf experiments
