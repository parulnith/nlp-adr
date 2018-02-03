# NLP for Adverse Drug Reaction mining

Final subsimmion for final project of "Classification approaches for Social Media Text",Summer Semester 2017,University of Potsdam.
Exploring different features for ADR mining (classify wheter a tweet contains ADR mention or not).

Code: 
- LSA
- paragraph2vec
- CNN as features extractor
- ADR lexicon score
- Sentiment Lexicon scores
- negation handling

Data :
- ADR twitter data downloaded with script at http://diego.asu.edu/downloads/twitter_annotated_corpus/
- sentiment lexicon : https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html
- ADR lexicon by [Sarker](#references)  : http://diego.asu.edu/downloads/publications/ADRMine/ADR_lexicon.tsv
- pre-trained word embeddings by [Sarker](#references) : http://diego.asu.edu/Publications/ADRMine.html
- pre-trained SSWE by [Tang](#references) : https://www.microsoft.com/en-us/research/people/dutang/

This repo contains as well a report for the experiments and an example of running the code.

## INSTALL

Download or clone the git repo https://github.com/sgarda/nlp-adr/

## RUN EXPERIMENTS

$ cd nlp-adr/code

$ python main.py

## REQUIREMENTS

- Pyhton (>= 3)
- pandas (>= 0.19)
- scikit-learn (>= 0.19) 
- gensim (>= 2.2) 
- keras (>= 2.0) 

## REFERENCES

Sarker A, Gonzalez G; Portable Automatic Text Classification for Adverse Drug Reaction Detection via Multi-corpus Training, Journal of Biomedical Informatics, 2015 Feb;53:196-207. doi: 10.1016/j.jbi.2014.11.002. Epub 2014 Nov 8. (resources for feature extraction for this task can be found at: http://diego.asu.edu/Publications/ADRClassify.html)

D. Tang, F. Wei, N. Yang, M. Zhou, T. Liu, and B. Qin. Learning sentiment-specific word embedding for twitter sentiment classification. In Proceedings of the 52nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1555–1565, Baltimore, Maryland, June2014. Association for Computational Linguistics. URL
1429, 2014.





