# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 14:09:20 2022

@author: Gloria
"""

# =============================================================================
# Text Data Processing and Preparation in Python
# =============================================================================

# =============================================================================
# 1. Remove html tags 
# =============================================================================
from bs4 import BeautifulSoup

example_string = "<body>     Hi! I'm Maria. :) Link to bio: https://maria.com/info.html . I live in London - my address is: Main street 31, 1000. I love wandering around the neighbourhood! 💕    <body>"

soup = BeautifulSoup(example_string) # create a "Beautiful Soup" object, which represents the document as a nested data structure
print(soup.get_text())

clean_text = soup.get_text() # html tags are removed

# =============================================================================
# 2. Remove intervals in the begin and end of a string
# =============================================================================

clean_text.strip()
clean_text.rstrip()
clean_text.lstrip()

clean_text = clean_text.strip()

# =============================================================================
# 3. Remove URLs
# =============================================================================
from urlextract import URLExtract
# How to install: 1.Open "Anaconda prompt". 2.Then type: pip install urlextract

extractor = URLExtract() # create an URLExtract object which will be used to extract URLs from the text
urls = extractor.find_urls(clean_text)
print(urls) #  
clean_text = clean_text.replace(urls[0], " ") # clean the text from URLs

# =============================================================================
# 4. Expand contractions
# =============================================================================

import contractions
# How to install: 1.Open "Anaconda prompt". 2.Then type: pip install contractions

contractions.fix("isn't")
contractions.fix("I'm")

clean_text = contractions.fix(clean_text) 

# =============================================================================
# 5. Extract emoticons/emoji
# =============================================================================

import emot
# How to install: 1.Open "Anaconda prompt". 2.Then type: pip install emot

emot_obj = emot.core.emot() # Create an "emot" object
emot_obj.emoji(clean_text) # search for emoji
emot_obj.emoticons(clean_text) # search for emoticons

# 💡 NB! URLs and other symbols in the text can impact the search for emoticons
emot_obj.emoticons(example_string) # search for emoticon

# =============================================================================
#  6. Apply tokenization
# =============================================================================

# 1) Use split()

# Word tokenization
tokenized1 = clean_text.split() # by default splits by intervals - punctuation is retained

# Sentence tokenization
import re # Python library for working with regex 
sentences = re.compile('[.!?] ').split(clean_text) 
# with compile() we create a regex object - we create our "pattern". Then, we apply a split() method inherent to "regex" objects (different from the built-in split()).
# https://docs.python.org/3/library/re.html#re-objects

# 2) Use NLTK

# Word tokenization
from nltk.tokenize import word_tokenize, sent_tokenize 
tokenized2 = word_tokenize(clean_text) # punctuation is a separate token

# 💡 NB! Words with an apostrophe are also split with word_tokenize()
word_tokenize("I'm Maria.") 

# Sentence tokenization
sent_tokenize(clean_text)

# 3) Use keras
from keras.preprocessing.text import text_to_word_sequence 
# How to install: 1.Open "Anaconda prompt". 2.Then type: pip install keras

tokenized3 = text_to_word_sequence(clean_text, lower = False) # punctuation is removed

# 💡 NB! Words with an apostrophe are retained as they are.
text_to_word_sequence("I'm Maria.", lower = False)

# =============================================================================
# 7. Apply POS tagging
# =============================================================================

# Apply POS tagging  directly after tokenization, before other text cleaning procedures and stop words removal! 

import nltk
nltk.download('averaged_perceptron_tagger') # https://www.kaggle.com/datasets/nltkdata/averaged-perceptron-tagger
nltk.download('tagsets')

postags = nltk.pos_tag(tokenized2) # provide a list of tokens here

"hi".isalpha() 
".".isalpha() 

# Remove punctuation from the list with postags
postags_clean = []
for pair in postags:
    if pair[0].isalpha() == True:
        postags_clean.append(pair)

nltk.help.upenn_tagset() # list of tagsets (tagsets meaning) 
# 💡 NB! POS taggers are not 100% accurate! Choose a tagger trained on data in domain which is close to the domain of data under analysis.
 
# =============================================================================
# 8. Lowercase the text
# =============================================================================

"HI".lower()
tokenized2_lower = list(map(lambda x: x.lower(), tokenized2))

# =============================================================================
# 9. Remove digits and special characters
# =============================================================================

tokenized2_lower_clean = []
for word in tokenized2_lower:
    if word.isalpha() == True:
        tokenized2_lower_clean.append(word)

# =============================================================================
# 10. Remove stop words
# =============================================================================
from nltk.corpus import stopwords
nltk.download('stopwords')

stopwords_list = stopwords.words('english')
len(stopwords_list) # 179
# 💡 NB! Always check the list of stop words before removing them!

# Remove the stop words
tokenized2_final = []
for word in tokenized2_lower_clean:
    if word not in stopwords_list:
        tokenized2_final.append(word) # 9 words are excluded

# 💡 If you want to save code space -> use list comprehension instead:
tokenized2_final1 = [word for word in tokenized2_lower_clean if word not in stopwords_list]
del tokenized2_final1

# How to add a new word to the list of stop words?
stopwords_list.append("hi")

# How to remove a word from the list of stop words?
stopwords_list.remove('not')

# =============================================================================
# 11. Apply stemming/lemmatization
# =============================================================================

# Apply Stemming
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("english") # create a "stemmer" object
print(stemmer.stem("parking"))
print(stemmer.stem("history"))

tokenized2_stemmed = []
for word in tokenized2_final:
    word = stemmer.stem(word)
    tokenized2_stemmed.append(word)

# Apply Lemmatization
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet') 

lemmatizer = WordNetLemmatizer() # create a "Lemmatizer" object
lemmatizer.lemmatize("better") # pos tag is not provided

# 💡 Lemmatization needs POS tags:
lemmatizer.lemmatize("better", pos ="a")

# 💡 The same word may be noun or a verb
lemmatizer.lemmatize("parking", pos ="v")
lemmatizer.lemmatize("parking", pos ="n")
 
postags_clean[0]
postags_clean[0][1] # access the POS tag
postags_clean[0][1][0] # access the first character in the POS tag
   
for pair in postags_clean:
    if pair[1][0].lower() in ['n','v','a','r','s']:
        lemmatized_word = lemmatizer.lemmatize(pair[0], pos = pair[1][0].lower())
        print(lemmatized_word)

# "am" and "is" changed to "be"    
