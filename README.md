# NLP-Analysis

Im Repository befindet sich das IU Projekt zur NLP-Analyse von unstrukturierten Daten.
Die nlp-pipeline.py beinhaltet den Quellcode zur Analyse.

data_path muss hierbei den Dateipfad zur Datenquelle beinhalten.
Die Datenquelle besteht aus ~220 Tickets und ist in der public_tickets_final.csv zu finden.

Das Skript bestimmt im ersten Durchlauf die optimale Themenanzahl für LDA.
Hiernach wird der User aufgefordert die gewünschte Themenzahl in der Konsole einzugeben und mit "Enter" zu bestätigen.
Im Anschluss werden beide Pipelines (BoW - LDA - TF-IDF & TF-IDF - LSA) durchlaufen und die Ergebnisse präsentiert.

Das Skript benötigt eine Themenrange, die per default auf min. 2 und max. 10 eingestellt ist.
Die Variablen min_topics & max_topics können entsprechend angepasst werden.
Eine höhere Range geht hierbei mit einer höheren Trainingszeit einher.
