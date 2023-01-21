# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 16:54:04 2022

@author: Gloria
"""

# =============================================================================
# Text Data Vectorization
# =============================================================================
import pandas as pd

string1 = "the movie is ok"
string2 = "the beginning was boring...very boring"
string3 = "the plot was very good"

mydata = [string1,string2, string3] # save in one list

# # Example of normalization importance (the impact of document length)
# string1 = "the movie is ok  the movie is ok the movie is ok the movie is ok"
# string2 = "the beginning was boring...very boring"
# string3 = "the movie is ok"
# mydata = [string1,string2, string3]

# =============================================================================
# I. Apply Binary Vectorization
# =============================================================================

from sklearn.feature_extraction.text import CountVectorizer

# Create a vectorizer
vectorizer = CountVectorizer(binary = True) 

# Fit the vectorizer (create the model)
X = vectorizer.fit_transform(mydata) # "compressed" view of the sparse data matrix (more memory-efficient)
data_matrix = X.todense() # check-out the matrix

# Our vocabulary:
feature_names = vectorizer.get_feature_names()

# Turn to DataFrame
data_matrix_binary = pd.DataFrame(X.todense(), columns = feature_names)

# =============================================================================
# II. Apply Count Vectorization (absolute)
# =============================================================================

# Create a vectorizer
vectorizer = CountVectorizer() #  create a vectorizer

# Fit
X = vectorizer.fit_transform(mydata) 
data_matrix_count = pd.DataFrame(X.todense(), columns = vectorizer.get_feature_names())

# =============================================================================
# III. Apply Count Vectorization (relative)
# =============================================================================

from sklearn.feature_extraction.text import TfidfVectorizer

# Create a vectorizer
vectorizer = TfidfVectorizer(use_idf=False, norm="l1")

# Fit
X = vectorizer.fit_transform(mydata)
data_matrix_count_rel = pd.DataFrame(X.todense(), columns = vectorizer.get_feature_names())

# =============================================================================
# IV. Apply TF-IDF Vectorization 
# =============================================================================

from sklearn.feature_extraction.text import TfidfVectorizer

# Create a vectorizer
vectorizer = TfidfVectorizer()
# vectorizer = TfidfVectorizer(norm = None) # apply without normalization

# Fit
X = vectorizer.fit_transform(mydata)
data_matrix_tf_idf = pd.DataFrame(X.todense(), columns = vectorizer.get_feature_names())


# =============================================================================
# V. Vectorize new observations
# =============================================================================

new_string = ["the movie is overall good"]

# Vectorize
new_string_transformed = vectorizer.transform(new_string) # NB! Use .transform() for new observations!
# NB! check out the difference between fit_transform() and transform() -> https://stackoverflow.com/questions/23838056/what-is-the-difference-between-transform-and-fit-transform-in-sklearn

data_matrix_tf_idf_new = pd.DataFrame(new_string_transformed.todense(), columns = vectorizer.get_feature_names())

# =============================================================================
# VI. Calculate text similarity
# =============================================================================

string1 = "the movie is ok"
string2 = "the beginning was boring"
string3 = "the plot was good"
string4 = "the cat ate my homework yesterday"
string5 = "my cat and dog are good"

mydata = [string1,string2, string3, string4, string5] # put in one list

# 1. Fit the vectorizer
vectorizer = TfidfVectorizer() # initialize the vectorizer
X = vectorizer.fit_transform(mydata) # fit
data_matrix_tf_idf = pd.DataFrame(X.todense(), columns = vectorizer.get_feature_names())

# 2. Import the "cosine_similarity" function
from sklearn.metrics.pairwise import cosine_similarity
cosine_sim = cosine_similarity(X) # the output looks like a "correlation matrix"

# 3. Extract the most similar documents to: string2 = "the beginning was boring"
mydata[1]
cosine_sim[1] # vector of similarities for string2

# 4. Create a dataframe with similarity scores for string2
similarity_scores = pd.DataFrame({'dist':cosine_sim[1], 'strings':mydata}).sort_values(by = 'dist', ascending=False)

# ! Check out the similarity scores for string5. 

# 6. Compute similarity between two strings without using the "cosine_similarity" function
sum(data_matrix_tf_idf.iloc[3,:]*data_matrix_tf_idf.iloc[4,:]) # 0.2721742151126849 # similarity between string4 and string5
# ! Compare with the output from the "cosine_similarity" function.

# 7. Compute similarity for new texts
string6 = ['the movie is great'] #  new text
Y = vectorizer.transform(string6) # vectorize -> NB! Use .transform() for new observations!
cosine_sim_new = cosine_similarity(X,Y) 

# OR the manual calculation
data_matrix_tf_idf_new = pd.DataFrame(Y.todense(), columns = vectorizer.get_feature_names())
sum(data_matrix_tf_idf.iloc[0,:]*data_matrix_tf_idf_new.iloc[0,:]) #
sum(data_matrix_tf_idf.iloc[1,:]*data_matrix_tf_idf_new.iloc[0,:])

