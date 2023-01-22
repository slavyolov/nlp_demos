# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 12:24:38 2023

@author: FEBA
"""

# =============================================================================
# Calculate Mutual Information - An example
# =============================================================================

# Create an example corpus of text data
a = ['the', 'movie', 'is', 'ok']
b = ['the', 'beginning', 'was', 'boring', 'very', 'boring']
c = ['the', 'movie', 'was', 'very', 'good']

all_text = [a,b,c]

# Import modules
import nltk
from nltk.collocations import BigramCollocationFinder,TrigramCollocationFinder 

# Calculate PMI for bigrams

bigram_measures = nltk.collocations.BigramAssocMeasures()
finder = BigramCollocationFinder.from_documents(all_text)

finder.nbest(bigram_measures.pmi, 20)

# Extract PMI scores
scores = bigram_measures.pmi
bigram_collocations = {"_".join(bigram): pmi for bigram, pmi in finder.score_ngrams(scores)}
bigram_collocations


