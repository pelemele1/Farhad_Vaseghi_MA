
all: main

main: clean
		latex RichtigZitieren
		bibtex RichtigZitieren
		latex RichtigZitieren
		latex RichtigZitieren
		dvips RichtigZitieren
		ps2pdf RichtigZitieren.ps
#		make clean

clean:
		rm -f *.log *.aux *.blg *.out *.dvi *.ps *.dvi
		rm -f RichtigZitieren.bbl

