# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 21:08:10 2022

@author: Gloria
"""

# =============================================================================
# 1. Import data
# =============================================================================
import pandas as pd
appdata = pd.read_csv('app reviews.csv') # 288065 obs.

appdata.columns
apps = appdata.package_name.value_counts() # 395 apps

appdata.isna().sum() # check for NA values

# =============================================================================
# 2. Create the target variable - Transform the star rating to positive/negative labels
# =============================================================================

# Create a custom function for the development of the target variable
def label_sentiment(star_rating):
    '''
    A function for creating the target variable.

    Parameters
    ----------
    star_rating : int
        Star rating which will be transformed to a category. 1 or 2 - negative; 4 or 5 - positive; 3 - neutral.

    Returns
    -------
    str
        The function returns the sentiment category - positive, negative, neutral.

    '''
    if star_rating == 1 or star_rating == 2:
        rating = 'neg'
    elif star_rating == 4 or star_rating == 5:
        rating = 'pos'
    else:
        rating = 'neutral'
    
    return rating


# Apply the function to appdata['star'] column
appdata['sentiment'] = appdata['star'].apply(label_sentiment)
appdata['sentiment'].value_counts()
# pos        211621
# neg         53248
# neutral     23196

# => Most of the reviews are positive.

# Check for duplicates in the combination of review + sentiment category
appdata[['review','sentiment']].duplicated().sum() # 64678 obs.

# Remove the duplicated values
appdata = appdata[~appdata[['review','sentiment']].duplicated()]
appdata.sentiment.value_counts()
# pos        154722
# neg         48383
# neutral     20282

# =============================================================================
# 3. Sample development
# =============================================================================

# For simplicity, in the current experiment I will use only part of the sample.
# In this regard, I will randomly select 20000 observations from each class (pos/neg) of the target variable

pos_reviews = appdata[appdata['sentiment'] == 'pos'].sample(n=20000, random_state = 42,  replace=False)
neg_reviews = appdata[appdata['sentiment'] == 'neg'].sample(n=20000,random_state = 42, replace=False)
# NB! Don't forget to set the random state (if you want to reproduce the experiment!)

# Put the sampled data in one dataframe
all_reviews = pd.concat([pos_reviews, neg_reviews])
all_reviews[['review','sentiment']].duplicated().sum()

# =============================================================================
# 4. Split the dataset to train and test samples
# =============================================================================

# In the current experiment I will use the "validation set" approach.
# The train sample (75% of observations in "all_reviews" dataframe) will be used to develop the prediction model.
# The test sample (25% of observations in "all_reviews" dataframe) will be used to validate the prediction model on unseen data.

from sklearn.model_selection import train_test_split
train, test = train_test_split(all_reviews,train_size = 30000, random_state=42, stratify = all_reviews['sentiment'])
# NB! Don't forget to "stratify" the target variable - this means that the target distribution will remain the same in both train and test samples.

train.sentiment.value_counts()
test.sentiment.value_counts()

# =============================================================================
# 5. Extract features from the train set (extract features related to punctuation)
# =============================================================================

# NB! It is advisable to check for ? and ! after extraction and removal of emoticons.

# Create a custom function for feature extraction

def feature_extraction(text):
    '''
    
    Parameters
    ----------
    text : str
        Input raw text of the review.

    Returns
    -------
    newtext2 : str
        The review with added new features indicating the presence of ?/! or all letters in upper case.

    '''
    # Feature indicating whether the review is in UPPER case.
    if str(text).isupper() == True:
        newtext = str(text)+ ' ' + 'ALLCAPITALS'
    else:
        newtext = text
    
    # Feature indicating whether the text contains EXCLAMATION mark.
    if '!' in newtext:
        newtext1 = newtext + ' ' + 'HASEXCLAMATION'
    else:
        newtext1 = newtext
    
    # Feature indicating whether the text contains QUESTION mark.
    if '?' in newtext1:
        newtext2 = newtext1 + ' ' + 'HASQUESTION'
    else:
        newtext2 = newtext1
    
    return newtext2

# Test the function
feature_extraction("ARE YOU INSANE?!") # works
feature_extraction("i am happy")
feature_extraction("am i happy?")
feature_extraction("yes, i am!!!")

# Apply on the train sample
train['review_features_extracted'] = train['review'].apply(feature_extraction)

# =============================================================================
#  6. Text processing
# =============================================================================

# Create a custom function for text processing
# NB! You can add more text processing techniques in the function below.

# Load necessary libraries
import contractions
import re
from keras.preprocessing.text import text_to_word_sequence # ignore the warning
from nltk.corpus import stopwords
# nltk.download('stopwords')

# Create a list with stop words which will be used during text processing
stopwords_list = stopwords.words('english')
stopwords_list.remove('not')
stopwords_list.remove('very')

# Create the function for text processing

def text_processing(text):
    '''
    
    Parameters
    ----------
    text : str
        Input raw text (before text processing).

    Returns
    -------
    text_tokenized_clean2 : str
        Output tokenized text (list) after text processing and normalization.

    '''
    
    # 1. Expand contractions
    text = contractions.fix(text)
    
    # 2. Lowercase, tokenize and remove punctuation
    text_tokenized = text_to_word_sequence(text)
    
    # 3. Remove stop words
    text_tokenized_clean1 = []
    
    for word in text_tokenized:
        if word not in stopwords_list:
            text_tokenized_clean1.append(word)
            
    # 4. Remove digits and special characters (if such are left) 
    text_tokenized_clean2 = []
    
    for word in text_tokenized_clean1:
        if any(list(map(lambda x: x.isalpha(), word))): # leave only words containing at least one alpha character
            word = re.sub("[^a-zA-Z]", "", word)  # remove punctation and digits attached to alpha characters
            text_tokenized_clean2.append(word)
                
    return text_tokenized_clean2

text_processing('Hey this is a 123 test :))')
text_processing('poor.app')

# Apply on train sample
train['processed_text'] = train['review_features_extracted'].apply(lambda x: text_processing(x))

# =============================================================================
# 7. Create wordclouds - are there clear differences between POS and NEG reviews?
# =============================================================================

# Create a custom function which finds the most common words in different target categories
from collections import Counter

def find_common_words(text_column):
    '''
    
    Parameters
    ----------
    text_column : pandas column, lists
         A pandas column of reviews in which to find the most common words.

    Returns
    -------
    word_dict : Counter object
        A dictionary which contains words and their corresponding frequency.

    '''
    all_words = []
    
    for item in text_column:
        all_words.extend(item)
    
    word_dict = Counter()
    
    for word in all_words:
        word_dict[word] += 1
    
    return word_dict

# Find most common words separately for POS/NEG reviews
word_dict_pos = find_common_words(train[train['sentiment'] == 'pos']['processed_text'])
word_dict_neg = find_common_words(train[train['sentiment'] == 'neg']['processed_text'])

# Inspect the dictionary 
word_dict_pos.most_common(20)  # Top 20 most common words  

# Generate the wordclouds with most common words
# More about the wordcloud library - https://pypi.org/project/wordcloud/ ; https://amueller.github.io/word_cloud/generated/wordcloud.WordCloud.html

# How to install: 1.Open "Anaconda prompt" as an administrator ("run as administrator"). 2.Then type: conda install -c conda-forge wordcloud

from wordcloud import WordCloud
wc = WordCloud(background_color = 'white', width = 1000, height = 1000, random_state=42)
wc.generate_from_frequencies(word_dict_pos)
#wc.to_file("wordcloud.png")

# Plot in Spyder
import matplotlib.pyplot as plt
plt.imshow(wc,interpolation='bilinear')
plt.axis("off")
plt.show()

wc = WordCloud(background_color = 'black', width = 1000, height = 1000, random_state=42)
wc.generate_from_frequencies(word_dict_neg)

plt.imshow(wc,interpolation='bilinear')
plt.axis("off")
plt.show()

# =============================================================================
# 8. Text vectorization and feature selection
# =============================================================================
from sklearn.feature_extraction.text import CountVectorizer

# NB! In the current use case will be applied binary vectorization!

# Create a binary vectorizer and perform feature selection based on word frequency
vectorizer = CountVectorizer(binary = True, max_df = 0.9, min_df = 10) 

# How do we perform the feature selection based on frequency? 
# Answer: The max_df and min_df parameters in the CountVectorizer function can be used to remove the most frequent and most rare terms in the sample of data. In the code above are removed all words that are presented in more than 90% of the reviews and all words that are found in less than 10 reviews.

# Perform the vectorization (by applying fit_transform() on the train dataset)
X_train = vectorizer.fit_transform([' '.join(i) for i in train['processed_text']])  
feature_names = vectorizer.get_feature_names() # list of extracted features

# =============================================================================
# 9. Fit the Bernoulli Naive Bayes model on the train set
# =============================================================================
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,classification_report

NB_model = BernoulliNB() # the Bernoulli Naive Bayes model is suited for text data in binary representation 
NB_model.fit(X_train,train['sentiment']) # the model is fit on X_train (the vectorized dataset)!

# Evaluate performance on the train sample
predictions_train_set = NB_model.predict(X_train)

# Confusion matrix
cm = confusion_matrix(train['sentiment'], predictions_train_set, labels=['pos','neg'])
cmd = ConfusionMatrixDisplay(cm, display_labels=['pos','neg'])
cmd.plot()

# Classification report
print(classification_report(train['sentiment'], predictions_train_set))

# => Accuracy on train set: 81%

# =============================================================================
# 10. Evaluate the performance of the model on unseen data (the test set)
# =============================================================================

# Transform the test set accordingly (apply the same feature extraction and text processing techniques)
test['review_features_extracted'] = test['review'].apply(feature_extraction)
test['processed_text'] = test['review_features_extracted'].apply(lambda x: text_processing(x))
X_test = vectorizer.transform([' '.join(i) for i in test['processed_text']])  
# NB! It is very important to use .transform() on the test set in order to apply the same vectorization scheme!!

# Make predictions on the test set
predictions_test_set = NB_model.predict(X_test)

# Confusion matrix on test set
cm = confusion_matrix(test['sentiment'], predictions_test_set, labels=['pos','neg'])
cmd = ConfusionMatrixDisplay(cm, display_labels=['pos','neg'])
cmd.plot()

# Classification report on test set
print(classification_report(test['sentiment'], predictions_test_set))
# => Accuracy on test set: 80%
