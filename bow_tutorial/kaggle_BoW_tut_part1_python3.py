import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords # Import the stop word list
from sklearn.feature_extraction.text import CountVectorizer # for extracting the most common words
import numpy as np
from sklearn.ensemble import RandomForestClassifier
# I also have scipy installed, which sklearn depends on

train = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)

def review_to_words( raw_review ):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and 
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review, "html5lib").get_text() 
    #
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))                  
    # 
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words ))

clean_review = review_to_words( train["review"][0] )
print(clean_review)

num_review = train["review"].size
print("Cleaning and parsing the training set movie reviews...\n")
clean_train_reviews = []
for i in range( 0, num_review ):
    if( (i+1)%1000 == 0 ):
        print("Review %d of %d\n" % ( i+1, num_review ))
    clean_train_reviews.append( review_to_words( train["review"][i] ) )

print("Creating the bag of words...\n")

# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.  
vectorizer = CountVectorizer(analyzer = "word",   \
                              tokenizer = None,     \
                              preprocessor = None,  \
                              stop_words = None,    \
                              max_features = 5000)
 
# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of 
# strings.
train_data_features = vectorizer.fit_transform(clean_train_reviews)
train_data_features = train_data_features.toarray() # Numpy arrays are easy to work with, so convert the result to an array
print(train_data_features.shape)
print("Should be (25000, 5000). \n")

vocab = vectorizer.get_feature_names()
print(vocab)
# imported numpy on line 7
dist = np.sum(train_data_features, axis=0) # Sum up the counts of each vocabulary word
# For each, print the vocabulary word and the number of times it 
# appears in the training set
for tag, count in zip(vocab, dist):
    print(count, tag)

# now it's time for the random forest classifier! It's supervised learning - tree-based classifiers to make predictions
print("Training the random forest...")

# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators = 100)

# Fit the forest to the training set, using the bag of words as features and the sentiment labels as the response variable

forest = forest.fit( train_data_features, train["sentiment"] )

# Now it's time to test the test data. Downloaded testData.tsv from the website. Our task is to predict the sentiment label.
test = pd.read_csv("testData.tsv", header=0, delimiter="\t", \
                   quoting=3 )
clean_test_reviews = []
print("Cleaning and parsing the test set movie reviews...\n")
for i in xrange(0,num_reviews):
    if( (i+1) % 1000 == 0 ):
        print("Review %d of %d\n" % (i+1, num_reviews))
    clean_review = review_to_words( test["review"][i] )
    clean_test_reviews.append( clean_review )
# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()
# Use the random forest to make sentiment label predictions -- here is where the predictions happen!
result = forest.predict(test_data_features)
# Copy the results to a pandas dataframe with an "id" column and
# a "sentiment" column
output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )
# Use pandas to write the comma-separated output file
output.to_csv( "Bag_of_Words_model.csv", index=False, quoting=3 )

