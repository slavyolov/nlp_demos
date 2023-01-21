# -*- coding: utf-8 -*-
"""
Created on Fri May 20 18:14:33 2022

@author: Gloria
"""

# =============================================================================
# I. Import the scraped tweets
# =============================================================================
import dill
with open('scraped_tweets.pkl', 'rb') as f: # set your working directory!
    tweets = dill.load(f)  

# =============================================================================
# II. Clean the text data
# =============================================================================

# 1). Check for duplicates and NAs
tweets.columns
tweets.tweet.duplicated().sum() # 11
tweets.tweet.isna().sum()  # 0

# 2). Remove duplicated tweets
tweets = tweets[~tweets['tweet'].duplicated()]

# HINT: Sort the 'tweet' column and inspect the results. Are there still duplicates (after the removal above)?

# 3). Remove URLs
from urlextract import URLExtract
# How to install: 1.Open "Anaconda prompt". 2.Then type: pip install urlextract

extractor = URLExtract() # create an URLExtract object which will be used to extract URLs from the text

cleaned_tweets1 = [] # this list will be used to store the result
for tweet in tweets['tweet']:
    urls = extractor.find_urls(tweet) # returns a list of found URLs (or empty list if none are found)
    for url in urls:
        tweet = tweet.replace(urls[0], " ") # clean the text from URLs (replace them with space)
    cleaned_tweets1.append(tweet)

tweets['tweets_no_URL'] = cleaned_tweets1

# Now check for duplicates again
tweets['tweets_no_URL'].duplicated().sum() # 209

# Important observation: It turns out that there are more duplicated tweets (only the URL in the tweet was different).

# Filter out the duplicated tweets
tweets = tweets[~tweets['tweets_no_URL'].duplicated()] # 780 obs.

# 4). Apply several text processing techniques within one loop

import contractions
from keras.preprocessing.text import text_to_word_sequence # ignore the warning
# How to install: 1.Open "Anaconda prompt". 2.Then type: pip install keras
import re

cleaned_tweets_string = [] # this list will be used to store the result
cleaned_tweets_tokenized = [] # this list will be used to store the result

for tweet in tweets['tweets_no_URL']:
    
    # Expand contractions
    tweet = contractions.fix(tweet) # expand contractions

    # Lowercase, tokenize and remove punctuation
    tweet_tokenized = text_to_word_sequence(tweet)  # punctuation is automatically removed + the text is lowercased
    
    # Remove digits and special characters (if such are left)
    clean_tweet_tokenized = []
    for word in tweet_tokenized:
        if any(list(map(lambda x: x.isalpha(), word))): # leave only words containing at least one alpha character
            word = re.sub("[^a-zA-Z]", "", word)  # remove punctation and digits attached to alpha characters
            clean_tweet_tokenized.append(word)
     
    # Convert Back to string format
    clean_tweet_string = ' '.join(clean_tweet_tokenized)
    
    # Attach the processed strings to a new list
    cleaned_tweets_string.append(clean_tweet_string) # string version
    cleaned_tweets_tokenized.append(clean_tweet_tokenized) # tokenized version

# Test the expression above
word = "CTV's"  
list(map(lambda x: x.isalpha(), word))
any(list(map(lambda x: x.isalpha(), word)))
re.sub("[^a-zA-Z]", "", word)

word = "123"
list(map(lambda x: x.isalpha(), word))
any(list(map(lambda x: x.isalpha(), word)))
    
# 5). Add the cleaned tweets as new columns in the dataframe    
    
tweets['tweets_clean_str'] = cleaned_tweets_string 
tweets['tweets_clean_tokenized'] = cleaned_tweets_tokenized 

del cleaned_tweets_string,cleaned_tweets_tokenized, cleaned_tweets1

# 6). Inspect the results

print("Raw version: " + tweets['tweet'][1])
print("Clean version: " + tweets['tweets_clean_str'][1])

# 7). Clean the environment

del clean_tweet_string,clean_tweet_tokenized,extractor,f,tweet,tweet_tokenized,url,urls,word

# =============================================================================
# III. Explore the metadata
# =============================================================================

tweets.columns

# Tweet with most retweets
print(tweets[tweets['retweet_count'] == max(tweets.retweet_count)]['tweet'].values) 
tweets[tweets['retweet_count'] == max(tweets.retweet_count)]['tweet'].index

# Tweet with most likes
print(tweets[tweets['like_count'] == max(tweets.like_count)]['tweet'].values) 
tweets[tweets['like_count'] == max(tweets.like_count)]['tweet'].index

# =============================================================================
# IV. Explore the text - create wordclouds
# =============================================================================
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

# 1). Import list with stop words
stopwords_list = stopwords.words('english')
len(stopwords_list) # 179
# 💡 NB! Always check the list of stop words before removing them!

# How to remove a word from the list of stop words?
stopwords_list.remove('not')
stopwords_list.append('https')

# 2). Find most common words 
all_words = [] # this list stores all words (with repetition)
for item in tweets['tweets_clean_tokenized']:
    all_words.extend(item)

# Count how many times each word occurs (excluding the stop words)
from collections import Counter

word_dict = Counter()

for word in all_words:
    if word not in stopwords_list: # exclude the stop words
        word_dict[word] += 1

# Inspect the dictionary 
word_dict.most_common(10)  # Top 10 most common words      

# 3). Generate the wordcloud with most common words
# More about the wordcloud library - https://pypi.org/project/wordcloud/ ; https://amueller.github.io/word_cloud/generated/wordcloud.WordCloud.html

# How to install: 1.Open "Anaconda prompt" as an administrator ("run as administrator"). 2.Then type: conda install -c conda-forge wordcloud

from wordcloud import WordCloud
wc = WordCloud(background_color = 'black', width = 1000, height = 1000, random_state=42)
wc.generate_from_frequencies(word_dict)
wc.to_file("wordcloud.png")

# Plot in Spyder
import matplotlib.pyplot as plt
plt.imshow(wc,interpolation='bilinear')
plt.axis("off")
plt.show()

# =============================================================================
# V. Use key words to select specific groups of interest
# =============================================================================

# Select all tweets containg the word "trial"

selected_tweets = [] # store the result

for tweet in tweets['tweets_clean_tokenized']:
    if "trial" in tweet:
        selected_tweets.append(tweet)
        
# # ! Additonal info - Filter according to several keywords:
# keyword_list = ['controlling','jealous'] # put your key words here
# for tweet in tweets['tweets_clean_tokenized']:
#       if any([keyword in tweet for keyword in keyword_list]):
#           selected_tweets.append(tweet)       

# # Check the logic
# tweet = "controlling and angry"
# [keyword in tweet for keyword in keyword_list]
# any([keyword in tweet for keyword in keyword_list])           
        
# =============================================================================
# VI. Apply key phrases extraction
# =============================================================================
   
import nltk
from nltk.collocations import BigramCollocationFinder,TrigramCollocationFinder 

# How to use - https://www.nltk.org/howto/collocations.html - Pointwise mutual information - PMI
bigram_measures = nltk.collocations.BigramAssocMeasures()
trigram_measures = nltk.collocations.TrigramAssocMeasures()

finder = BigramCollocationFinder.from_documents(tweets['tweets_clean_tokenized'])
finder.apply_freq_filter(15)
finder.nbest(bigram_measures.pmi, 20)
    
finder = TrigramCollocationFinder.from_documents(tweets['tweets_clean_tokenized'])
finder.apply_freq_filter(10)
finder.nbest(trigram_measures.pmi,20)

# =============================================================================
# VII. Sentiment Analysis with the help of Lexicons
# =============================================================================

import pandas as pd

##### 1). Use AFINN - check out: https://pypi.org/project/afinn/
from afinn import Afinn 
# How to install: 1.Open "Anaconda prompt". 2.Then type: pip install afinn

afinn = Afinn()

# Apply on tweets
sentiment_scores_afinn = []
for tweet in tweets['tweets_clean_str']:
    sent_score = afinn.score(tweet)
    sentiment_scores_afinn.append(sent_score)
    
tweets['AFINN_scores'] = sentiment_scores_afinn

# HINT: You can use an ifelse statement and turn the numerical ratings into labels (positive/negative/neutral)

len(tweets[tweets['AFINN_scores'] >0]) # 217 POS
len(tweets[tweets['AFINN_scores'] <0]) # 333 NEG
len(tweets[tweets['AFINN_scores'] ==0]) # 230 NEU

##### 2). Use TextBLob - check out: https://textblob.readthedocs.io/en/dev/quickstart.html

from textblob import TextBlob

# Example output with textblob
test = TextBlob("Textblob is amazingly simple to use. What great fun!")
test.sentiment
test.sentiment.polarity
test.sentiment.subjectivity

# Example of an objective sentence
test = TextBlob("the johnny depp and amber heard defamation trial everything to know cnet")
test.sentiment.polarity # 0
test.sentiment.subjectivity  # objective 

# Apply on tweets
sentiment_scores_textblob = []
subjectivity_scores_textblob = []

for tweet in tweets['tweets_clean_str']:
    scores = TextBlob(tweet)
    sentiment_scores_textblob.append(scores.sentiment.polarity)
    subjectivity_scores_textblob.append(scores.sentiment.subjectivity)
    
tweets['TextBlob_polarity_scores'] = sentiment_scores_textblob
tweets['TextBlob_subjectivity_scores'] = subjectivity_scores_textblob

len(tweets[tweets['TextBlob_subjectivity_scores'] > 0.2]) # 464

##### 3). Use NRC - check out: https://pypi.org/project/NRCLex/

# More information: https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm

from nrclex import NRCLex
# How to install: 1.Open "Anaconda prompt". 2.Then type: pip install NRCLex 

text_scored = NRCLex('johnny depp was a controlling lover ex girlfriend testifies ellen barkin says johnny depp was a jealous and angry man even back in the s') 

text_scored.top_emotions
text_scored.top_emotions[0][0]
text_scored.raw_emotion_scores #  +1 when a word from a given category occurs 
# In the example above, 2 words fall into the category "anger" and "disgust"
text_scored.affect_frequencies # emotion_count/all_emotions_count

emotion_scores_NRC = []

for tweet in tweets['tweets_clean_str']:
    text_scored = NRCLex(tweet)
    emotions_list = list(map(lambda x: x[0],text_scored.top_emotions)) 
    emotion_scores_NRC.append(emotions_list)

tweets['emotion_tags'] = emotion_scores_NRC

# Count emotions
all_emotions = [] # this list stores all words (with repetition)
for item in tweets['emotion_tags']:
    all_emotions.extend(item)

emotion_count = Counter()

for emotion in all_emotions:
    emotion_count[emotion] += 1

# =============================================================================
# VIII. Sentiment Analysis with the help of pre-trained models 
# =============================================================================

from transformers import pipeline

# How to install?
# 1. pip install transformers
# 2. install Pytorch - https://pytorch.org/get-started/locally/ - follow the instructions
# Model used below: https://huggingface.co/siebert/sentiment-roberta-large-english
# Check out other models for sentiment analysis here: https://huggingface.co/models?language=en&sort=downloads&search=sentiment

sentiment_analysis = pipeline("sentiment-analysis", model="siebert/sentiment-roberta-large-english")
sentiment_analysis("I love this!")

sentiment_analysis("I hate this!")[0]['label']

sentiment_scores_SIEBERT = []

for tweet in tweets['tweets_clean_str']:
    text_scored = sentiment_analysis(tweet)
    sentiment_scores_SIEBERT.append(text_scored[0]['label'])

tweets['sentiment_scores_SIEBERT'] = sentiment_scores_SIEBERT
tweets['sentiment_scores_SIEBERT'].value_counts()

